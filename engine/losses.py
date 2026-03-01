import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(x, default: bool = False) -> bool:
    """Robust bool parser for cfg values coming from CLI --set."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(int(x) != 0)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("0", "false", "no", "off", "none", "null", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
    return bool(x)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return _GradReverse.apply(x)


class TACEOptimizationField(nn.Module):
    """
    Stable TACE loss for RootSAM-Pro (SEQUENCE+TBPTT safe).

    Key fixes:
      1) USE_TASK_UNCERTAINTY correctly supports "0"/"1" from CLI --set.
      2) If uncertainty disabled -> NO exp scaling, Total stays in sane range.
      3) SDF-exempt focal BCE uses denom >= 1.0 (no 0.01*HW blow-up).
      4) Pure-soil FP penalty (FULL) uses softplus(topk_logits) for stronger suppression.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = str(getattr(cfg, "ABLATION_MODE", "FULL"))

        # weights (legacy)
        self.l_seg = float(getattr(cfg, "L_SEG", 1.0))
        self.l_topo = float(getattr(cfg, "L_TOPO", 0.25))

        # uncertainty switch (robust)
        self.use_task_uncertainty = _as_bool(getattr(cfg, "USE_TASK_UNCERTAINTY", True), default=True)
        self.unc_clamp = float(getattr(cfg, "UNCERTAINTY_LOGVAR_CLAMP", 2.0))  # exp(2)=7.39 max scaling
        if self.use_task_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # seg, topo
        else:
            self.log_vars = None

        # focal params
        self.alpha = float(getattr(cfg, "ALPHA_FOCAL", 0.85))
        self.gamma = float(getattr(cfg, "GAMMA_FOCAL", 2.0))
        self.eps = 1e-6

        # soft clDice
        self.skel_iters = int(getattr(cfg, "TOPO_SKEL_ITER", 10))

        # soil dual constraint
        self.soil_lambda_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  # stronger than -4
        self.soil_lambda_max = float(getattr(cfg, "SOIL_LAMBDA_MAX", 30.0))
        self.soil_topk_ratio = float(getattr(cfg, "SOIL_TOPK_RATIO", 0.02))  # 2%

        # caps
        self.seg_cap = float(getattr(cfg, "SEG_LOSS_CAP", 50.0))

    def _ensure_1ch(self, x):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4 and x.size(1) != 1:
            x = x[:, 0:1]
        return x

    # -----------------------------
    # Soft clDice helpers
    # -----------------------------
    @staticmethod
    def _soft_erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _soft_dilate(img):
        return F.max_pool2d(img, (3, 3), stride=1, padding=1)

    def _soft_open(self, img):
        return self._soft_dilate(self._soft_erode(img))

    def _soft_skeletonize(self, img: torch.Tensor, iters: int) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        skel = torch.zeros_like(img)
        curr = img
        for _ in range(int(iters)):
            opened = self._soft_open(curr)
            delta = F.relu(curr - opened)
            skel = skel + F.relu(delta - skel * delta)
            curr = self._soft_erode(curr)
        return skel.clamp(0.0, 1.0)

    def _cldice_loss(self, pred_prob: torch.Tensor, gt_prob: torch.Tensor) -> torch.Tensor:
        skel_pred = self._soft_skeletonize(pred_prob, self.skel_iters)
        skel_gt = self._soft_skeletonize(gt_prob, self.skel_iters)
        tprec = (skel_pred * gt_prob).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + 1e-6)
        tsens = (skel_gt * pred_prob).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + 1e-6)
        cldice = (2 * tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cldice.mean()

    def _focal_bce_exempt(self, logits: torch.Tensor, gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """SDF-exempt focal BCE (fp32-safe, normalized, stable)."""
        prob = torch.sigmoid(logits).float().clamp(self.eps, 1.0 - self.eps)
        gt = gt.float()
        w = w.float().clamp(0.0, 1.0)

        pt = prob * gt + (1.0 - prob) * (1.0 - gt)
        focal = (self.alpha * gt + (1.0 - self.alpha) * (1.0 - gt)) * torch.pow(1.0 - pt, self.gamma)
        bce = -(gt * torch.log(prob) + (1.0 - gt) * torch.log(1.0 - prob))
        loss_pix = focal * bce * w

        denom = torch.clamp(w.sum(), min=1.0)  # ✅ stable denom
        out = loss_pix.sum() / denom
        if self.seg_cap > 0:
            out = torch.clamp(out, max=self.seg_cap)
        return out

    def forward(
        self,
        M_logits: torch.Tensor,
        M_gt: torch.Tensor,
        W_SDF: torch.Tensor | None = None,
        probes_prob: torch.Tensor | None = None,
        Pc_gt: torch.Tensor | None = None,
        Pt_gt: torch.Tensor | None = None,
        Ph_gt: torch.Tensor | None = None,
        F_t: torch.Tensor | None = None,
        F_prev: torch.Tensor | None = None,
        F_kin: torch.Tensor | None = None,
    ):
        device = M_logits.device
        dev_type = "cuda" if device.type == "cuda" else "cpu"

        M_gt = self._ensure_1ch(M_gt)
        M_logits = self._ensure_1ch(M_logits)
        if W_SDF is not None:
            W_SDF = self._ensure_1ch(W_SDF)

        gt = M_gt.float().clamp(0.0, 1.0)
        pred_prob = torch.sigmoid(M_logits).float()

        # --- seg ---
        if W_SDF is None:
            W_SDF = torch.ones_like(gt)
        else:
            W_SDF = W_SDF.to(device=device).float().clamp(0.0, 1.0)

        with torch.autocast(device_type=dev_type, enabled=False):
            loss_seg = self._focal_bce_exempt(M_logits.float(), gt.float(), W_SDF.float())

        # --- topo (always computed; you can gate by mode if you want) ---
        with torch.autocast(device_type=dev_type, enabled=False):
            loss_topo = self._cldice_loss(pred_prob.float(), gt.float())

        # --- total (uncertainty or fixed) ---
        if self.use_task_uncertainty and (self.log_vars is not None):
            s = self.log_vars.clamp(-self.unc_clamp, self.unc_clamp)
            total = torch.exp(-s[0]) * loss_seg + 0.5 * s[0] + torch.exp(-s[1]) * loss_topo + 0.5 * s[1]
        else:
            total = self.l_seg * loss_seg + self.l_topo * loss_topo

        # --- pure-soil fp dual constraint (FULL only) ---
        soil_fp = torch.zeros((), device=device, dtype=torch.float32)
        soil_lam = torch.zeros((), device=device, dtype=torch.float32)
        soil_topk = 0

        if self.mode == "FULL":
            gt_sum = gt.flatten(1).sum(dim=1)  # (B,)
            idx = torch.nonzero(gt_sum <= 0, as_tuple=False).view(-1)
            if idx.numel() > 0:
                soil_logits = M_logits.index_select(dim=0, index=idx).float().flatten(1)  # (N,HW)
                soil_topk = max(1, int(soil_logits.size(1) * max(1e-4, self.soil_topk_ratio)))
                topk_vals = torch.topk(soil_logits, k=soil_topk, dim=1, largest=True).values  # (N,k)

                # stronger penalty than sigmoid mean
                soil_fp = F.softplus(topk_vals).mean()

                soil_lam = F.softplus(grad_reverse(self.soil_lambda_raw)).clamp(0.0, self.soil_lambda_max)
                total = total + soil_lam * soil_fp

        if not torch.isfinite(total):
            # last defense
            total = (loss_seg + loss_topo).detach()

        loss_dict = {
            "Total": float(total.detach().item()),
            "Seg_Exempt": float(loss_seg.detach().item()),
            "Topo_clDice": float(loss_topo.detach().item()),
        }
        if self.mode == "FULL":
            loss_dict["SoilFP"] = float(soil_fp.detach().item()) if torch.is_tensor(soil_fp) else float(soil_fp)
            loss_dict["SoilLambda"] = float(soil_lam.detach().item()) if torch.is_tensor(soil_lam) else float(soil_lam)
            loss_dict["SoilTopK"] = int(soil_topk)

        return total, loss_dict
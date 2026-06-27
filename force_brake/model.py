import torch
import torch.nn as nn

# extend from nn.Module

class ForceBrakeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = 0.5
        self.eps = 1e-6

    def forward(self, force):
        # force: [batch, 3] -> [n, fx, fy]
        n = force[:, 0:1]
        f = force[:, 1:3]

        # amend normal force
        n = torch.relu(n)
        n_safe = torch.clamp(n, min=self.eps)

        # amend tangetial force
        # total tangetial force
        f_tan = torch.norm(f, dim=1, keepdim = True) + self.eps

        # max tangetial force
        max_f_tan = self.mu * n_safe

        # scale
        # max_f_tan/f_tan and 1 for min
        scale = torch.where(
            f_tan > max_f_tan,
            max_f_tan / f_tan,   # larger than max,
            torch.ones_like(f_tan)
        )

        f = f * scale

        return torch.cat([n, f], dim=1)

def test(num_test = 1000, batch_size = 32):

    # fallback to CPU if no GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ForceBrakeNN().to(device)
    model.eval()

    for i in range(num_test):
        force = torch.randn(
            batch_size,
            3,
            device = device,
            requires_grad=True
        )

        out = model(force)

        n = out[:, 0:1]
        f = out[:, 1:3]

        f_tan = torch.norm(f, dim=1, keepdim = True)
        max_f = model.mu * n

        # constrain check
        if torch.any(f_tan > max_f + 1e-6):
            print(f"[FAIL] constraint violation at iter {i}")
            return False

        # backward check
        loss = torch.relu(f_tan - max_f).mean()
        loss.backward()

        if torch.isnan(force.grad).any() or torch.isinf(force.grad).any():
            print(f"[FAIL] NaN/Inf grad at iter {i}")
            return False

        force.grad.zero_()
        print(f"[INFO] iter {i} passed")

    print("[PASS] all tests passed")
    return True

if __name__ == "__main__":
    test()

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import gc
from abc import ABC, abstractmethod

# --- IMPORT MODELS ---
from sparse_k_transformer import Sparse2Trans, Sparse3Trans
from baseline_models import MPNN3Body, MPNNModel, DeepSetBaseline
from k_set_transformer import Full2Trans, Full3Trans
from k_transformer import Full2TransSeq, Full3TransSeq

# --- 1. TASK MODULARITY ---

class ParamCalculator:
    @staticmethod
    def get_params(model):
        """Counts trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def solve_d(model_class_fn, target_params, tolerance=0.05):
        """
        Iteratively finds the d_model that results in a model size closest to target_params.
        """
        best_d = 16
        min_diff = float('inf')

        for d in range(16, 256, 4):
            try:
                model = model_class_fn(d)
                p = ParamCalculator.get_params(model)
                diff = abs(p - target_params)

                if diff < min_diff:
                    min_diff = diff
                    best_d = d

                if p > target_params * (1 + tolerance) and diff > min_diff:
                    break
            except Exception as e:
                print(f"  [ParamCalculator] d={d} failed: {e}")
                continue

        return best_d

class PhysicsTask(ABC):
    @abstractmethod
    def generate_batch(self, batch_size, device):
        """Returns (inputs [B, N, 2], targets [B, 2])"""
        pass

    @abstractmethod
    def name(self):
        pass

# --- TASKS ---

class QuadraticVertexTask(PhysicsTask):
    """
    Predict the vertex (h, k) of a quadratic y = a(x-h)^2 + k from sampled points.
    3-body because curvature (second derivative) requires 3 points.
    """
    def __init__(self, num_points):
        self.num_points = num_points

    def name(self):
        return f"Quadratic Vertex (N={self.num_points})"

    def generate_batch(self, batch_size, device):
        N = self.num_points
        a = torch.rand(batch_size, 1, device=device) * 1.5 + 0.5
        h = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0
        k = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0

        local_x = torch.rand(batch_size, N, device=device) * 4.0 - 2.0
        x = local_x + h
        y = a * (x - h)**2 + k

        points = torch.stack([x, y], dim=2)
        target = torch.cat([h, k], dim=1)
        return points, target

class ThirdMomentTask(PhysicsTask):
    """
    Predict the third central moments (m3_x, m3_y) of a 2D point cloud.
    3-body because the third central moment is a degree-3 polynomial of the multiset
    involving irreducible triple products.
    """
    def __init__(self, num_points):
        self.num_points = num_points

    def name(self):
        return f"Third Moment (N={self.num_points})"

    def generate_batch(self, batch_size, device):
        N = self.num_points

        # Mixture of 2 Gaussians with random asymmetry => controllable skewness
        offset_x = torch.rand(batch_size, 1, device=device) * 2.0 + 0.5
        offset_y = torch.rand(batch_size, 1, device=device) * 2.0 + 0.5
        w = torch.rand(batch_size, 1, device=device) * 0.6 + 0.1
        sigma = torch.rand(batch_size, 1, device=device) * 0.3 + 0.2

        mask = (torch.rand(batch_size, N, device=device) < w).float()

        noise_x = torch.randn(batch_size, N, device=device) * sigma
        noise_y = torch.randn(batch_size, N, device=device) * sigma
        x = noise_x + mask * offset_x - (1 - mask) * offset_x
        y = noise_y + mask * offset_y - (1 - mask) * offset_y

        mean_x = x.mean(dim=1, keepdim=True)
        mean_y = y.mean(dim=1, keepdim=True)
        m3_x = ((x - mean_x) ** 3).mean(dim=1)
        m3_y = ((y - mean_y) ** 3).mean(dim=1)

        points = torch.stack([x, y], dim=2)
        # Scale targets to keep MSE in a comparable range to other tasks
        target = torch.stack([m3_x, m3_y], dim=1) / 5.0

        return points, target

class CircleCenterTask(PhysicsTask):
    """
    Predict the center (cx, cy) of a circle from noisy points on an arc.
    3-body because 3 non-collinear points uniquely determine a circle (circumcenter).
    Arc sampling prevents the centroid shortcut.
    """
    def __init__(self, num_points):
        self.num_points = num_points

    def name(self):
        return f"Circle Center (N={self.num_points})"

    def generate_batch(self, batch_size, device):
        N = self.num_points

        cx = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0
        cy = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0
        R  = torch.rand(batch_size, 1, device=device) * 1.5 + 0.5

        # Sample from a random arc (not full circle) to prevent centroid shortcut
        arc_start  = torch.rand(batch_size, 1, device=device) * 2 * math.pi
        arc_length = torch.rand(batch_size, 1, device=device) * math.pi + math.pi / 2
        theta = arc_start + torch.rand(batch_size, N, device=device) * arc_length

        x = cx + R * torch.cos(theta)
        y = cy + R * torch.sin(theta)

        # Small noise relative to radius
        x = x + torch.randn_like(x) * 0.05
        y = y + torch.randn_like(y) * 0.05

        points = torch.stack([x, y], dim=2)
        target = torch.cat([cx, cy], dim=1)
        return points, target

class CubicInflectionTask(PhysicsTask):
    """
    Predict the inflection point (p, q) of a cubic y = a(x-p)^3 + q from sampled points.
    3-body because the second derivative (needed to locate the inflection) requires
    the second finite difference — an irreducible 3-point computation. Unlike the
    quadratic (constant curvature), the cubic's curvature varies linearly with x,
    making this strictly harder for 2-body models.
    """
    def __init__(self, num_points):
        self.num_points = num_points

    def name(self):
        return f"Cubic Inflection (N={self.num_points})"

    def generate_batch(self, batch_size, device):
        N = self.num_points

        a = torch.rand(batch_size, 1, device=device) * 1.5 + 0.5
        sign = torch.randint(0, 2, (batch_size, 1), device=device).float() * 2 - 1
        a = a * sign

        p = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0
        q = torch.rand(batch_size, 1, device=device) * 4.0 - 2.0

        # Narrower range than quadratic to keep y-values reasonable
        local_x = torch.rand(batch_size, N, device=device) * 3.0 - 1.5
        x = local_x + p
        y = a * (x - p) ** 3 + q

        points = torch.stack([x, y], dim=2)
        target = torch.cat([p, q], dim=1)
        return points, target

# --- 2. EXPERIMENT CONTROLLER ---

class ExperimentSuite:
    def __init__(self, task, device=None):
        self.task = task
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.static_data = {}

    def register(self, name, model, static_input=None):
        model = model.to(self.device)
        self.models[name] = model

        if static_input is not None:
            if isinstance(static_input, (list, tuple)):
                self.static_data[name] = [x.to(self.device) for x in static_input]
            else:
                self.static_data[name] = static_input.to(self.device)
        else:
            self.static_data[name] = None

        params = ParamCalculator.get_params(model)
        print(f"  Registered '{name}' | Params: {params:,} | d_model: {model.d_model}")

    def cleanup(self):
        """Move all models/data off GPU and free CUDA memory."""
        for name in list(self.models.keys()):
            self.models[name].cpu()
            del self.models[name]
        for name in list(self.static_data.keys()):
            v = self.static_data[name]
            if isinstance(v, list):
                for t in v:
                    del t
            elif v is not None:
                del v
        self.models.clear()
        self.static_data.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, steps=1000, batch_size=32):
        print(f"\n--- {self.task.name()} ---")
        opts = {k: optim.Adam(v.parameters(), lr=1e-3) for k, v in self.models.items()}
        loss_fn = nn.MSELoss()
        history = {k: [] for k in self.models}

        for step in range(steps):
            x, y = self.task.generate_batch(batch_size, self.device)

            for name, model in self.models.items():
                opts[name].zero_grad()

                extra = self.static_data[name]
                if extra is None:
                    pred = model(x)
                elif isinstance(extra, list):
                    pred = model(x, *extra)
                else:
                    pred = model(x, extra)

                loss = loss_fn(pred, y)
                loss.backward()
                opts[name].step()
                history[name].append(loss.item())

            if step % 200 == 0:
                losses_str = " | ".join(f"{n}: {h[-1]:.4f}" for n, h in history.items())
                print(f"  Step {step}: {losses_str}")

        return history

    def plot(self, history):
        plt.figure(figsize=(12, 6))
        for name, losses in history.items():
            w = min(50, len(losses))
            smoothed = [sum(losses[i:i+w])/w for i in range(max(1, len(losses)-w+1))]
            plt.plot(smoothed, label=name)
        plt.yscale('log')
        plt.title(f'{self.task.name()}: Training Curves')
        plt.ylabel('MSE Loss')
        plt.xlabel('Step')
        plt.legend(fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- 3. HELPERS ---

def build_debruijn_edges(N, degree=2):
    src, dst = [], []
    for i in range(N):
        for r in range(degree):
            src.append(i)
            dst.append((i * degree + r) % N)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

def build_suite(task, N, target_params=45000):
    """Create an ExperimentSuite with all models registered.
    O(N^3) full models (SetTrans-3, SeqTrans-3) are skipped when N > 48
    because their forward pass allocates O(N^3 * d_model) memory.
    """
    suite = ExperimentSuite(task)
    include_cubic = N <= 48

    print(f"\nBuilding suite for: {task.name()}")
    print(f"  Target params: ~{target_params:,}")
    if not include_cubic:
        print(f"  Skipping O(N^3) full models (N={N} > 48)")

    # --- Solve balanced dimensions ---
    d_deep  = ParamCalculator.solve_d(lambda d: DeepSetBaseline(d), target_params)
    d_set2  = ParamCalculator.solve_d(lambda d: Full2Trans(d), target_params)
    d_seq2  = ParamCalculator.solve_d(lambda d: Full2TransSeq(d, N), target_params)
    d_sp2   = ParamCalculator.solve_d(lambda d: Sparse2Trans(d), target_params)
    d_sp3   = ParamCalculator.solve_d(lambda d: Sparse3Trans(d), target_params)
    d_mpnn2 = ParamCalculator.solve_d(lambda d: MPNNModel(d), target_params)
    d_mpnn3 = ParamCalculator.solve_d(lambda d: MPNN3Body(d), target_params)

    if include_cubic:
        d_set3 = ParamCalculator.solve_d(lambda d: Full3Trans(d, N), target_params)
        d_seq3 = ParamCalculator.solve_d(lambda d: Full3TransSeq(d, N), target_params)

    # --- Graph structures ---
    db_src, db_dst = build_debruijn_edges(N)
    num_edges = db_src.shape[0]
    rand_src = torch.randint(0, N, (num_edges,))
    rand_dst = torch.randint(0, N, (num_edges,))
    rand_triplets = torch.randint(0, N, (num_edges, 3))

    # --- Register models ---
    # 1-body baseline
    suite.register("DeepSet (1-body)", DeepSetBaseline(d_deep))

    # Set-Transformers (permutation invariant)
    suite.register("SetTrans-2 (O(N^2))", Full2Trans(d_set2))
    if include_cubic:
        suite.register("SetTrans-3 (O(N^3))", Full3Trans(d_set3, N))

    # Seq-Transformers (position aware)
    suite.register("SeqTrans-2 (O(N^2))", Full2TransSeq(d_seq2, N))
    if include_cubic:
        suite.register("SeqTrans-3 (O(N^3))", Full3TransSeq(d_seq3, N))

    # Sparse models (edges/triplets baked in)
    suite.register("DeBruijn-2 (Struct)", Sparse2Trans(d_sp2, db_src, db_dst))
    suite.register("Sparse-2 (Rand)", Sparse2Trans(d_sp2, rand_src, rand_dst))
    suite.register("Sparse-3 (Rand)", Sparse3Trans(d_sp3, rand_triplets))

    # MPNN baselines (edges/triplets at forward time)
    suite.register("MPNN-2 (Rand)", MPNNModel(d_mpnn2),
                   static_input=torch.stack([rand_src, rand_dst], dim=1))
    suite.register("MPNN-3 (Rand)", MPNN3Body(d_mpnn3),
                   static_input=rand_triplets)

    return suite

# --- 4. RUNNER ---
if __name__ == "__main__":
    N = 128
    TARGET_PARAMS = 45000
    STEPS = 1000

    tasks = [
        QuadraticVertexTask(N),
        ThirdMomentTask(N),
        CircleCenterTask(N),
        CubicInflectionTask(N),
    ]

    for task in tasks:
        suite = build_suite(task, N, TARGET_PARAMS)
        history = suite.run(steps=STEPS)
        suite.plot(history)
        suite.cleanup()
        del suite

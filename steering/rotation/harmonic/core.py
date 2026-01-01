"""Forward degree-1+2 sky manifold and live orthogonal component replacement."""
from __future__ import annotations
import hashlib
import re
import numpy as np

MODEL = 'mistralai/Mistral-Large-Instruct-2411'
REVISION = 'ba78820945ae22361b0274cf0ae6d696c967c1a4'
SEED = 20260917


def layer_groups(metadata):
    # Highest patchable hidden state: the final one is post-final-norm and is
    # never hooked here. Defaults to Mistral-Large's 87 for historical bundles.
    top=metadata.get('max_hidden_state',87)
    groups=metadata.get('layer_groups',[[l] for l in metadata['layers']])
    for group in groups:
        if not group or group!=sorted(set(group)) or not set(group).issubset(metadata['layers']):
            raise ValueError('Layer groups must be sorted, unique, nonempty subsets of fitted layers')
        if min(group)<1 or max(group)>top:
            raise ValueError(f'Residual hooks support hidden states 1-{top}; the final state is normalized')
    return groups


def group_label(group):
    if len(group)==1:return group[0]
    if group==list(range(group[0],group[-1]+1)):return f'{group[0]}-{group[-1]}'
    return ','.join(map(str,group))


def trace_records(trace):
    return trace if isinstance(trace,list) else [trace]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unit(u):
    u = np.asarray(u, dtype=np.float64)
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError('Zero sky direction')
    return u / n


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def harmonics(u):
    """Real l=1,2 basis with E_uniform[phi phi.T]=I; constant is separate.

    Axis convention follows archive (sin Dec, cos Dec sin RA, cos Dec cos RA).
    Choice of polar axis/basis signs cannot change the fitted OLS subspace.
    """
    x, y, z = np.moveaxis(unit(u), -1, 0)
    return np.stack([np.sqrt(3)*x, np.sqrt(3)*y, np.sqrt(3)*z,
                     np.sqrt(15)*x*y, np.sqrt(15)*x*z, np.sqrt(15)*y*z,
                     np.sqrt(15)/2*(x*x-y*y), np.sqrt(5)/2*(3*z*z-1)], axis=-1)


def fit_forward(u, z, degree=2):
    phi = harmonics(u)[..., :3 if degree == 1 else 8]
    design = np.column_stack([np.ones(len(phi)), phi])
    coef, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Rank-deficient harmonic design')
    return coef[0], coef[1:].T


def basis(W):
    q, s, _ = np.linalg.svd(W, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-10))
    return q[:, :rank], s


class Chart:
    def __init__(self, path=None, **arrays):
        if path is not None:
            with np.load(path) as d:
                arrays = {k: d[k].astype(np.float64) for k in d.files}
        for k in ['mu', 'sigma', 'V', 'pca_mean', 'b', 'W']:
            setattr(self, k, arrays[k])
        self.Q, self.singular_values = basis(self.W)
        self.gram = self.V @ self.V.T
        # Correct archived float32 loadings' tiny orthogonality error while
        # preserving the complete standardized complement of the PCA rowspace.
        self.lift = np.linalg.solve(self.gram, self.V) * self.sigma[None, :]

    def scores(self, h):
        return ((np.asarray(h, dtype=np.float64)-self.mu)/self.sigma-self.pca_mean) @ self.V.T

    def replace(self, h, target):
        z = self.scores(h)
        wanted = harmonics(target) @ self.W.T
        dz = ((wanted-(z-self.b)) @ self.Q) @ self.Q.T
        delta = dz @ self.lift
        return np.asarray(h, dtype=np.float64)+delta

    def trace(self, before, after, target):
        z0, z1 = self.scores(before), self.scores(after)
        wanted = harmonics(target) @ self.W.T @ self.Q
        error = (z1-self.b) @ self.Q-wanted
        dz = z1-z0
        outside = dz-(dz @ self.Q) @ self.Q.T
        dh = np.asarray(after, dtype=np.float64)-before
        std_delta = dh/self.sigma
        pca_lift = np.linalg.solve(self.gram, dz) @ self.V
        return dict(target_relative_error=float(np.linalg.norm(error)/max(np.linalg.norm(wanted),1e-12)),
                    target_error_norm=float(np.linalg.norm(error)),
                    pc_complement_change_norm=float(np.linalg.norm(outside)),
                    outside_retained_pcs_change_norm=float(np.linalg.norm(std_delta-pca_lift)),
                    standardized_delta_norm=float(np.linalg.norm(std_delta)),
                    delta_norm=float(np.linalg.norm(dh)),
                    delta_relative_norm=float(np.linalg.norm(dh)/np.linalg.norm(before)),
                    before_p=((z0-self.b) @ self.Q).tolist(), after_p=((z1-self.b) @ self.Q).tolist(),
                    desired_p=wanted.tolist())


def display(name):
    return name.replace('_', ' ').removesuffix(' constellation')


def parse_answer(text, catalog):
    aliases = {a.casefold(): c['id'] for c in catalog for a in [c['name'],display(c['name'])]}
    for c in catalog:
        if display(c['name']) == 'Crux':
            aliases['southern cross'] = c['id']
    cleaned = text.strip().lstrip('*\"\' ').casefold()
    cleaned = re.sub(r'^(?:(?:the\s+)?constellation(?:\s+of)?\s+|it\s+is\s+)', '', cleaned)
    cleaned = re.sub(r'^the\s+', '', cleaned)
    match = re.match(r'('+'|'.join(re.escape(a) for a in sorted(aliases,key=len,reverse=True))+r')\b', cleaned)
    if match is None:
        return None
    tail = cleaned[match.end():].lstrip('*\"\' ')
    if re.match(r'^(?:or|and|/|is\s+not|isn\x27t)\b', tail):
        return None
    return aliases[match.group()]

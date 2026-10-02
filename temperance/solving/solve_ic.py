"""
Frobenius (leading-order) analysis near x = 0 for the stellar
oscillation system (Dziembowski 1971 / Unno et al. formulation):

  x dy1/dx = (Vg - 2) y1 + (1 - Vg/eta) y2 - Vg y3           (11)
  x dy2/dx = [l(l+1) - eta*A] y1 + (A - 1) y2 + eta*A y3     (12)
  x dy3/dx = y3 + y4                                           (13)
  x dy4/dx = -A*U y1 - U*(Vg/eta) y2
             + [l(l+1) + U*(A-2) + U*Vg] y3 + 2*(1-U) y4     (14)

PHYSICAL PARAMETERS
-------------------
At the stellar centre:  U = 3,  Vg = 0.

With Vg=0 the stellar-structure identity A*eta = Vg*l gives A=0, so
the matrix M_ode completely decouples into two 2x2 blocks:

    (y1, y2) block:   eigenvalues  s = l-1  and  s = -l-2
    (y3, y4) block:   eigenvalues  s = l-1  and  s = -l-2  (at U=3!)

Both blocks share the root s = l-1, so the null space of (M - (l-1)I)
is GENUINELY 2-DIMENSIONAL, spanned by two clean eigenvectors:

    v_12 = [1/(l+1), 1, 0, 0]^T     (from the y1-y2 block)
    v_34 = [0, 0, 1/(l-2), 1]^T     (from the y3-y4 block, l≠2)

COINCIDENCE AT U=3
------------------
The (y3,y4) block eigenvalue at general U is
    s_{34} = (3 - 2U ± sqrt((2l+1)^2 - 4*(2U-1))) / 2
At U=3 this simplifies to (3 - 6 ± (2l+1)) / 2, giving l-1 and -l-2,
exactly matching the (y1,y2) block.  This is the physical reason the
null space becomes 2D: U=3 is the condition for equal central exponents.

BASIS CHOICE
------------
The natural basis with the requested normalisations is:

    basis_A : c1=1, c3=0  —  set by v_12 normalised to c1=1
    basis_B : c1=1, c3=1  —  v_34 (normalised c3=1) + beta * v_12 (c1=1)
                              where beta = -v_34[0] = -1/(l+1) * ... 
                              In fact v_34[0]=0, so basis_B = v_34 shifted
                              by the c1=1 normalisation of v_12 times
                              (1 - 0) = 1, giving simply v_34 + 1*v_12(c1=1).
"""

import sympy as sp

l = sp.Symbol('l', positive=True)


def _build_Mode_physical():
    """
    Return M_ode evaluated at the physical central values U=3, Vg=0, A=0.
    eta drops out completely (no Vg or A terms survive).
    """
    U_val, Vg_val, A_val = 3, 0, 0
    return sp.Matrix([
        [Vg_val - 2,          1 - Vg_val,       -Vg_val,                              0],
        [l*(l+1) - 0,         A_val - 1,         0,                                   0],
        [0,                   0,                 1,                                   1],
        [0,                   0,                 l*(l+1) + U_val*(A_val-2),           2*(1-U_val)],
    ])


def frobenius_basis_physical(ell, verbose=True):
    """
    Frobenius basis at U=3, Vg=0 (stellar centre).

    Parameters
    ----------
    ell : int or sympy expression
        Harmonic degree l.

    Returns
    -------
    dict with keys:
        's'       : the shared indicial exponent  s = l-1
        'basis_A' : [c1,c2,c3,c4] with c1=1, c3=0
        'basis_B' : [c1,c2,c3,c4] with c1=1, c3=1
        'beta'    : admixture coefficient (= 1, since v_34 has c1=0 exactly)
        'v_12'    : raw null vector from (y1,y2) block, normalised c1=1
        'v_34'    : raw null vector from (y3,y4) block, normalised c3=1
    """
    M = _build_Mode_physical().subs(l, ell)
    s = sp.simplify(ell - 1)

    M_s = sp.simplify(M - s * sp.eye(4))

    null = M_s.nullspace()
    if len(null) != 2:
        raise ValueError(
            f"Expected 2D null space at l={ell}, U=3, Vg=0; got {len(null)}D.\n"
            "Check that l is symbolic or a positive integer ≠ 2 (l=2 gives c4=0/0 in v_34)."
        )

    # Identify which vector lives in (y1,y2) and which in (y3,y4)
    # by checking which components are nonzero.
    va, vb = [sp.simplify(v) for v in null]

    def _in_12_block(v):
        return sp.simplify(v[2]) == 0 and sp.simplify(v[3]) == 0

    if _in_12_block(va):
        v_12_raw, v_34_raw = va, vb
    elif _in_12_block(vb):
        v_12_raw, v_34_raw = vb, va
    else:
        raise ValueError("Could not identify block structure of null vectors.")

    # Normalise
    v_12 = sp.simplify(v_12_raw / v_12_raw[0])   # c1 = 1
    v_34 = sp.simplify(v_34_raw / v_34_raw[2])   # c3 = 1

    # basis_A : pure (y1,y2) block — already has c1=1, c3=0
    basis_A = v_12

    # basis_B : c1=1, c3=1.
    # v_34 has c1=0 exactly, so adding beta*v_12(c1=1) with beta=1 gives c1=1.
    beta = sp.simplify(1 - v_34[0])   # = 1 - 0 = 1  exactly
    basis_B = sp.simplify(v_34 + beta * v_12)

    if verbose:
        print("=" * 65)
        print("Frobenius basis at U=3, Vg=0  (stellar centre)")
        print("=" * 65)
        print(f"\n  l = {ell},  U = 3,  Vg = 0,  A = 0")
        print(f"  Shared regular exponent:  s = l-1 = {s}")
        print()
        print("Why 2D null space?")
        print("  At U=3 the (y1,y2) and (y3,y4) blocks both have")
        print("  eigenvalue l-1, so M-(l-1)I has rank 2, not 3.")
        print()
        print("v_12  (y1-y2 block null vector, c1=1):")
        _pprint_vec(v_12)
        print()
        print("v_34  (y3-y4 block null vector, c3=1):")
        _pprint_vec(v_34)
        print()
        print(f"  beta = 1 - v_34[c1] = {beta}  (v_34 has c1=0 exactly)")
        print()
        print("basis_A  [c1=1, c3=0]  =  v_12:")
        _pprint_vec(basis_A)
        print()
        print("basis_B  [c1=1, c3=1]  =  v_34 + beta * v_12  =  v_34 + v_12:")
        _pprint_vec(basis_B)
        print()
        print("General regular solution near x=0:")
        print("  y(x) = alpha * x^(l-1) * basis_A  +  gamma * x^(l-1) * basis_B")
        print("       = x^(l-1) * [alpha * basis_A + gamma * basis_B]")
        print("  =>  y1 ~ (alpha + gamma) x^(l-1),   y3 ~ gamma x^(l-1)")
        print("=" * 65)

    return {
        's': s,
        'basis_A': basis_A,
        'basis_B': basis_B,
        'beta': beta,
        'v_12': v_12,
        'v_34': v_34,
    }


def _pprint_vec(v):
    for i, ci in enumerate(v):
        print(f"  c{i+1} = ", end='')
        sp.pprint(sp.factor(sp.nsimplify(ci)), use_unicode=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 1. Symbolic in l ──────────────────────────────────────────────────────
    print("\n── SYMBOLIC (l kept symbolic) ───────────────────────────────────")
    r_sym = frobenius_basis_physical(l, verbose=True)

    # ── 2. Numerical example l=2 ──────────────────────────────────────────────
    # Note: at l=2 the v_34 formula has c4 = 1/(l-2) -> diverges.
    # This signals a logarithmic (Frobenius resonance) solution for l=2;
    # a power-series solution alone doesn't exist for y3,y4 at l=2, U=3.
    print("\n── NUMERICAL l=2 ────────────────────────────────────────────────")
    r_l2 = frobenius_basis_physical(2, verbose=True)
    print("Float values:")
    for name, key in [('basis_A', 'basis_A'), ('basis_B', 'basis_B')]:
        print(f"  {name}: {[float(ci) for ci in r_l2[key]]}")

    # ── 3. Cross-checks ───────────────────────────────────────────────────────
    print("\nCross-checks (l=2):")
    M2 = _build_Mode_physical().subs(l, 2)
    s2 = sp.Integer(1)  # s = l-1 = 1 at l=2

    for name, v_key in [('v_12', 'v_12'), ('v_34', 'v_34')]:
        resid = sp.simplify((M2 - s2*sp.eye(4)) * r_l2[v_key])
        ok = all(sp.simplify(r) == 0 for r in resid)
        print(f"  (M-s*I)*{name} = {list(resid)}  {'✓' if ok else '✗ FAIL'}")

    diff = sp.simplify(r_l2['basis_B'] - r_l2['beta']*r_l2['v_12'] - r_l2['v_34'])
    ok = all(sp.simplify(d) == 0 for d in diff)
    print(f"  basis_B - beta*v_12 - v_34 = {list(diff)}  {'✓' if ok else '✗ FAIL'}")

    bA, bB = r_l2['basis_A'], r_l2['basis_B']
    print(f"  basis_A: c1={bA[0]}, c3={bA[2]}  (want 1,0)  "
          f"{'✓' if bA[0]==1 and bA[2]==0 else '✗'}")
    print(f"  basis_B: c1={bB[0]}, c3={bB[2]}  (want 1,1)  "
          f"{'✓' if bB[0]==1 and bB[2]==1 else '✗'}")

    # ── 4. Warning for l=2 ───────────────────────────────────────────────────
    print()
    print("Note on l=2:")
    print("  The v_34 null vector has c4 = 1/(l-2), which diverges at l=2.")
    print("  At l=2, U=3, Vg=0 the (y3,y4) block has a repeated eigenvalue")
    print("  with a 1D eigenspace — a logarithmic (resonant) Frobenius solution")
    print("  y3 ~ x^(l-1) * (c3 + c3' * ln x) is required instead.")
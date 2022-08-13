include('Axioms/RBI001-0.ax').
cnf(left_identity, axiom, mult(e, X) = X).
cnf(right_identity, axiom, mult(X, e) = X).
cnf(lp, hypothesis, meet(e, join(undr(X, Y), undr(Y, X))) = e).
% ^\
cnf(meet_undr, hypothesis, undr(meet(x, y), z) != join(undr(x, z), undr(y, z))).

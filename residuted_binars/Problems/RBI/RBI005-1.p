include('Axioms/RBI001-0.ax').
cnf(lattice_distributivity, axiom, meet(X, join(Y, Z)) = join(meet(X, Y), meet(X, Z))).
cnf(join_over, hypothesis, over(join(x, y), z) = join(over(x, z), over(y, z))).
cnf(meet_undr, hypothesis, undr(meet(x, y), z) = join(undr(x, z), undr(y, z))).
cnf(undr_join, hypothesis, undr(x, join(y, z)) != join(undr(x, y), undr(x, z))).

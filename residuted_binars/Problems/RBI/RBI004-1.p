include('Axioms/RBI001-0.ax').
cnf(join_undr, hypothesis, undr(join(X, Y), Z) != meet(undr(X, Z), undr(Y, Z))).

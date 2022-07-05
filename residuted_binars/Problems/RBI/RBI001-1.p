include('Axioms/RBI001-0.ax').
cnf(undr_meet, hypothesis, undr(X, meet(Y, Z)) != meet(undr(X, Y), undr(X, Z))).

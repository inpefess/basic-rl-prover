include('Axioms/RBI001-0.ax').
cnf(meet_over, hypothesis, over(meet(X, Y), Z) != meet(over(X, Z), over(Y, Z))).

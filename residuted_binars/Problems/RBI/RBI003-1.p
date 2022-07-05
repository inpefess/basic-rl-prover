include('Axioms/RBI001-0.ax').
cnf(over_join, hypothesis, over(X, join(Y, Z)) != meet(over(X, Y), over(X, Z))).

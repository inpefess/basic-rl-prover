include('Axioms/RBI001-0.ax').
% /v
cnf(over_join, hypothesis, over(x, join(y, z)) != meet(over(x, y), over(x, z))).

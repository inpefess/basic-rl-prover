include('Axioms/RBI001-0.ax').
% ^/
cnf(meet_over, hypothesis, over(meet(x, y), z) != meet(over(x, z), over(y, z))).

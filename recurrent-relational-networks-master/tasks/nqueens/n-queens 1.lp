% DOMAIN
#const n=4. square(1..n,1..n).
% GENERATE
0 { queen(X,Y) } 1 :- square(X,Y).
% TEST
:- queen(X,Y), queen(X',Y'), X < X', X'- X = |Y'- Y|.
:- queen(X, Y), queen(X, Y'), Y != Y'.
:- queen(X, Y), queen(X', Y), X!=X'.


:- Q = #count {X, Y: queen(X, Y)}, Q != n.
#show queen/2.
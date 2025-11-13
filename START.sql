START
DECLARE n, i, isPrime
FOR n :arrow_left: 2 TO 100
    SET isPrime :arrow_left: TRUE
    FOR i :arrow_left: 2 TO n - 1
        IF n MOD i = 0 THEN
            SET isPrime :arrow_left: FALSE
            EXIT FOR
        END IF
    END FOR
    IF isPrime = TRUE THEN
        OUTPUT n
    END IF
END FOR
END
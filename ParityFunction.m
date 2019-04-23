X_train = [   1   , -1,   1;
        1   ,  1,   -1;
        -1  ,  1,   1;
        1   ,  1,   1;
        -1  , -1,  -1];
Y_train = [ 1;
            -1;
            -1;
            1;
            1];


[W, B] = create([3,4,2,1], 0.1, 0.01, X_train, Y_train)

X_test = [1 ,  -1 , 1;
            -1, -1, 1];
Y_test = [1; -1];

y_hat = feed(X_test, W, B)

error = epoch_error(Y_test, y_hat);
printf("the training error is %f\n", error);
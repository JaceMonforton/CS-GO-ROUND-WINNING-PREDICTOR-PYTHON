knn = KNeighborsClassifier(n_jobs=4)
# knn.fit(x_train_scaled, y_train)

# clf = RandomizedSearchCV(knn, param_grid, n_jobs=4, n_iter=3, verbose=2, cv=3)
# clf.fit(x_train_scaled, y_train)
# knn = clf.best_estimator_

# score = knn.score(x_test_scaled, y_test)


# #Random forest classifier
# forest = RandomForestClassifier(n_jobs=4)
# # forest.fit(x_train_scaled, y_train)
# score = forest.score(x_test_scaled, y_test)
# print(score)

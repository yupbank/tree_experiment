from openml import config, study, tasks, runs, extensions
from sklearn import compose, impute, metrics, pipeline, preprocessing, tree, ensemble
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def clf_from_params(**kwargs):
    return pipeline.make_pipeline(
        compose.make_column_transformer(
            (impute.SimpleImputer(), extensions.sklearn.cont),
            (preprocessing.OneHotEncoder(handle_unknown='ignore'), extensions.sklearn.cat),
            ),
        ensemble.RandomForestClassifier(**kwargs)
        )

benchmark_suite = study.get_suite('OpenML-CC18')

#handpick_tasks = [99,]
handpick_tasks = [
10093, 125920, 146800, 146821, 14952, 14970, 167120, 167141, 219, 29, 31, 3560, 
## 3904, 
43, 6, 9946, 9964, 9978, 10101, 125922, 146817, 146822, 14954, 15, 167121, 18, 22, 3, 32, 3573, 3913, 45, 7592, 9952, 9971, 9981, 11, 14, 146819, 146824, 14965, 16, 167124, 2074, 23, 3021, 3481, 37, 
## 3917, 
49, 99, 9957, 9976, 9985, 12, 146195, 146820, 146825, 14969, 167119, 167140, 2079, 28, 3022, 3549, 
## 3902, 
## 3918, 
53, 9910, 9960, 9977]

splitters = ['best', 'random', 'second_best', 'worst']
max_depths = [1, 5, 10, 15]
print('task_id,splitter,max_depth,dataset,mean_accuracy, std_accuracy,n_fold')
for task_id in handpick_tasks: # iterate over all tasks
#for task_id in handpick_tasks: # iterate over all tasks
    task = tasks.get_task(task_id) # download the OpenML task
    #X, y = task.get_X_and_y() # get the data (not used in this example)
    for splitter in splitters:
        for max_depth in max_depths:
            clf = clf_from_params(splitter=splitter, max_depth=max_depth)
            run_res = runs.run_model_on_task(clf, task) # run classifier on splits given by the task
            score = run_res.get_metric_fn(metrics.accuracy_score) # compute and print the accuracy score
            print(f'{task_id},{splitter},{max_depth},{task.get_dataset().name},{score.mean()},{score.std()},{"|".join(map(str, score))}')
    #run.publish()
    #run_ids.append(run.id)

#benchmark_study = study.create_study( # create a study to share the set of results
#alias='splitter',
#name="Tree splitter test",
#description="An study reporting results of a decision tree with different splitter.",
#run_ids=run_ids,
#benchmark_suite=benchmark_suite.id
#)
#benchmark_study.publish()
#print(f"Results are stored at {benchmark_study.openml_url}")

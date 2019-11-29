import sqlite3
import re
from itertools import combinations 
import shutil
from operator import itemgetter
from functools import reduce

from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

from utils import *
from utils.discretizer import Discretizer
from utils.exercise_sample_indexer import ExerciseSampleIndexer
from utils.itemset_indexer import ItemSetIndexer
from utils.pattern_clustering import PatternClustering


class PURE():
    def __init__(self, discretization_bin_count=8, pattern_mining_measure='PFPM',  
                 clustering_distance_metric='jaccard', clustering_affinity='average', 
                 attributeList = {
                    'name':['duration', 'distance', 'calorie',
                            'weekday', 'exerciseCalssId', 'exerciseTime', 'isExercise'],
                    'type':['continuous', 'continuous', 'continuous',
                        'discrete', 'discrete', 'discrete', 'discrete']
                }, 
                items = [
                    ['isExercise'], ['exerciseClassId'], 
                    ['exerciseClassId', 'duration'], 
                    ['exerciseClassId', 'distance'],
                    ['exerciseClassId', 'exerciseTime'],
                    ['exerciseClassId', 'calorie']
                ]):

        self.attributeList = attributeList
        self.items = items
        self.itemList = ['_and_'.join(item) for item in self.items]
        self.dctzer = Discretizer(self.attributeList, bin_count=discretization_bin_count)
        self.record_idxer = ExerciseSampleIndexer(self.itemList)
        self.pattern_mining_measure = pattern_mining_measure

        self.pattern_idxer = None
        self.pclust_samples = None
        self.clustering_distance_metric = clustering_distance_metric
        self.clustering_affinity = clustering_affinity

        self.__mode = 'train'
    
    def set_mode(self, mode):
        assert mode in ['train', 'test']
        self.__mode = mode

    def preprocess_exercise_records(self, exercise_records):
        if self.__mode == 'train':
            exercise_records = self.dctzer.fit_transform(exercise_records)
        elif self.__mode == 'test': 
            exercise_records = self.dctzer.transform(exercise_records)
        return exercise_records

    def construct_exercise_transactions(self, user_list, exercise_records, output_path='PFPM/input/'):
        # convert attributes to items
        for item in self.items:
            if len(item) == 1:
                continue
            exercise_records['_and_'.join(item)] = self._create_combination(exercise_records, item)
        
        # Before fitting, enlarge the feature space for column 'isExercise'
        exercise_records.loc[-1] = [
            0 if col=='isExercise' else x for (col, x) in zip(exercise_records.columns, exercise_records.loc[0])
        ]
        exercise_records.index = exercise_records.index + 1
        exercise_records = exercise_records.sort_index()

        # Item Encoding
        self.record_idxer.fit(exercise_records[self.itemList])

        # After fitting, remove this row
        exercise_records = exercise_records.drop(exercise_records.index[len(exercise_records)-1])

        # Transaction Generation
        exercise_transactions = []
        with tqdm(total=len(user_list)) as pbar:
            for userID in user_list:
                pbar.set_description("Processing user {:8s} ".format(str(userID)))
                
                data = self._dateMatching(exercise_records[exercise_records['userId'] == int(userID)])
                time_window_list = self._get_time_window_list(data)   

                for dt in time_window_list:  
                    file_path = os.path.join(output_path, '{}.{:4d}-{:02d}.txt'.format(userID, dt.year, dt.month))
                    exercise_transactions.append({
                        'user': userID,
                        'time_window': '{:4d}-{:02d}'.format(dt.year, dt.month),
                        'file_path': file_path
                    })
                    self._write_file(self._fetch_sequence_in_window(data, dt), file_path)
                pbar.update(1)
        return exercise_transactions

    def extract_exercise_patterns(self, exercise_transactions, 
                                  output_path='PFPM/output/', file_size_limit=10*10**9,
                                  args={'minper':1, 'maxper':10, 'minavgper':1, 'maxavgper':7}):
        if self.pattern_mining_measure == 'PFPM':
            self.minper, self.maxper = args['minper'], args['maxper']
            self.minavgper, self.maxavgper = args['minavgper'], args['maxavgper']
        elif self.pattern_mining_measure == 'FPM':
            self.minavgper, self.maxavgper = args['support'], 31
            
        exercise_patterns = []
        with tqdm(total=len(exercise_transactions)) as pbar:
            for trx in exercise_transactions:
                input_path = trx['file_path']
                userID, time_window, _ = trx['file_path'].split('/')[-1].split('.')
                pbar.set_description("Processing file {} ".format(input_path))

                file_path = os.path.join(output_path, '{}.{}.txt'.format(userID, time_window))
                exercise_patterns.append({
                    'user': userID,
                    'time_window': time_window,
                    'file_path': file_path
                })

                if self.pattern_mining_measure == 'PFPM':
                    self._run_PFPM(input_path, file_path, self.minper, self.maxper, self.minavgper, self.maxavgper)    
                elif self.pattern_mining_measure == 'FPM':
                    self._run_FPM(input_path, file_path, args['support'])
                pbar.update(1)
            
                for ptn in exercise_patterns:
                    input_path = ptn['file_path']
                    try:
                        file_size = os.stat(input_path).st_size
                    except OSError:
                        continue
                    if file_size > file_size_limit:
                        os.remove(input_path)

        return exercise_patterns

    def encode_exercise_patterns(self, exercise_patterns):
        pattern_data = {'userID': [], 'time_window': [], 'patterns': [], 
                        'periodicity_min': [], 'periodicity_max': [], 'periodicity_mean': []}
        with tqdm(total=len(exercise_patterns)) as pbar:
            for ptn in exercise_patterns:
                input_path = ptn['file_path']
                userID, time_window, _ = ptn['file_path'].split('/')[-1].split('.')
                pbar.set_description("Processing file {} ".format(input_path))

                try:
                    with open(input_path, 'r') as f:  
                        content = list(filter(lambda line: '#SUP: 1\t' not in line, f.readlines()))
                        
                        patterns = [line[:line.find('#')].strip() for line in content]
                        if self.pattern_mining_measure == 'PFPM':
                            periodicities = list(map(lambda line: float(line.split('#AVGPER: ')[-1]), content))
                        elif self.pattern_mining_measure == 'FPM':
                            periodicities = list(map(lambda line: float(line.split('#SUP: ')[-1]), content))
                        
                except IOError:
                    print('No such file or directory: "{}"'.format(input_path))
                    pbar.update(1)
                    continue
                    
                codes = self.pattern_idxer.encode(patterns)
                if len(codes) > 0:
                    pattern_data['userID'].append(userID)
                    pattern_data['time_window'].append(time_window)
                    pattern_data['patterns'].append(codes)
                    pattern_data['periodicity_min'].append(min(periodicities))
                    pattern_data['periodicity_max'].append(max(periodicities))
                    pattern_data['periodicity_mean'].append(sum(periodicities)/len(periodicities))

                    # if userID in rec:
                    #     rec[userID].append(time_window)
                    # else:
                    #     rec[userID] = [time_window]

                pbar.update(1)

        exercise_patterns_encoded = pd.DataFrame.from_dict(pattern_data)
        return exercise_patterns_encoded#, rec
    
    def _generate_pattern_vectors(self, exercise_patterns_encoded):  
        X = exercise_patterns_encoded['patterns'].tolist()
        
        pattern_vectors = np.zeros((len(X), self.pattern_idxer.size), dtype=bool)
        for i in range(len(X)):
            pattern_vectors[i, X[i]] = True
        
        return pattern_vectors.astype(bool)
    
    def prepare_pattern_vectors(self, exercise_patterns, occur_threshold=3, is_sqlite_solver=False, ret_info=False):    
        if self.pattern_idxer is None:
            self.fit_pattern_indexer(exercise_patterns, occur_threshold, is_sqlite_solver)

        if self.__mode == 'train':
            #if self.pclust_samples is None:
            self.pclust_samples = exercise_patterns_encoded = self.encode_exercise_patterns(exercise_patterns)
        elif self.__mode == 'test':
            exercise_patterns_encoded = self.encode_exercise_patterns(exercise_patterns)
        
        pattern_vectors = self._generate_pattern_vectors(exercise_patterns_encoded)
        if ret_info:
            return exercise_patterns_encoded, pattern_vectors
        else:
            return pattern_vectors
        
    
    def cluster_pattern_vectors(self, pattern_vectors, n_jobs=-1, pre_dispatch='1.5*n_jobs', temp_folder=None):
        assert self.__mode == 'train'
        
        if self.clustering_distance_metric == 'jaccard':
            distance_matrix = self.prepare_pattern_distances(pattern_vectors, n_jobs, pre_dispatch, temp_folder)
            self.pclust = PatternClustering(affinity=self.clustering_affinity, metric=self.clustering_distance_metric)
            self.pclust.fit(distance_matrix=distance_matrix)
        else:
            self.pclust = PatternClustering(affinity=self.clustering_affinity, metric=self.clustering_distance_metric)
            self.pclust.fit(pattern_vectors)
        
        self.pclust.show_elbow(title='Clusters Number Selection')
    
    def get_user_representations(self, exercise_patterns, params=None, n_jobs=-1): 
        # e.g., params = {'threshold': 10, 'criterion':'maxclust'}   
        
        if params is not None:
            self.pclust.threshold = params['threshold']
            self.pclust.criterion = params['criterion']
        
#         if self.__mode == 'train':
#             info = self.pclust_samples            
#             info['cluster'] = self.pclust.get_clusters()
            
#         elif self.__mode == 'test':
#             model_samples = (self._generate_pattern_vectors(self.pclust_samples), self.pclust.get_clusters())
#             info, pattern_vectors = self.prepare_pattern_vectors(exercise_patterns, ret_info=True)
#             info['cluster'] = self.pclust.predict(model_samples, pattern_vectors, n_jobs)
        model_samples = (self._generate_pattern_vectors(self.pclust_samples), self.pclust.get_clusters())
        info, pattern_vectors = self.prepare_pattern_vectors(exercise_patterns, ret_info=True)
        distance_to_clusters = self.pclust.predict(model_samples, pattern_vectors, n_jobs)
        
        feat_cols = ['userID', 'time_window', 'periodicity_min', 'periodicity_max', 'periodicity_mean']
        for i in range(len(distance_to_clusters)):
            info['cluster_{}'.format(i)] = [1.-d for d in distance_to_clusters[i, :]]
            feat_cols += ['cluster_{}'.format(i)]
        return info[feat_cols]
    
    def show_cluster_chracteristics(self, threshold=10, criterion='maxclust', n_characteristics=20):
        self.pclust.threshold = threshold
        self.pclust.criterion = criterion
        
        info = self.pclust_samples
        codes = self.pclust.get_clusters()
        
        info['cluster'] = codes
        
        for c in np.unique(codes):
            cls_data = info[(info['cluster'] == c)] #& (result['period'] == period)]
            print('[Cluster {}]\t{} periodic user(s)'.format(c, cls_data.shape[0]))
            self._count_patterns(cls_data['patterns'], n=n_characteristics)
            print('='*50)

    def prepare_pattern_distances(self, pattern_vectors, n_jobs=-1, pre_dispatch='1.5*n_jobs', temp_folder=None, save=False):
        def jaccard(X, i):
            ret = []
            u = X[i]
            for j in range(i, len(X)):
                v = X[j]
                nonzero = np.bitwise_or(u != 0, v != 0)
                unequal_nonzero = np.bitwise_and((u != v), nonzero)
                a = np.double(unequal_nonzero.sum())
                b = np.double(nonzero.sum())
                ret.append((a / b) if b != 0 else 0)
            return ret

        distance_matrix = []
        print('Number of tasks: {}'.format(len(pattern_vectors)))
        out = Parallel(n_jobs=n_jobs, verbose=5, pre_dispatch=pre_dispatch, 
                    temp_folder=temp_folder)(delayed(jaccard)(pattern_vectors, i) for i in range(len(pattern_vectors))) 
        if save:
            self._save_distance_matrix(out, path='pclust_dist')

        comb = combinations([_ for _ in range(len(pattern_vectors))], 2) 
        return squareform(np.array([out[i][j-i] for (i, j) in comb]))
    
    def _save_distance_matrix(self, out, path='dist'):
        def write_list_to_file(myList, output_path, var_name):
            with open(output_path, 'w') as f:
                f.write("{}='''".format(var_name))
                f.write('\n'.join([str(x) for x in myList]))
                f.write("'''.split('\\n')")

        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)

        for i in range(len(out)):
            write_list_to_file(out[i], os.path.join(path, 'dist_{}.py'.format(i)), 'dist_{}'.format(i))
        with open(os.path.join(path, '__init__.py'), 'w') as f:
            f.write('out = []\n')
            for i in range(len(out)):
                f.write('from dist_{} import dist_{}\n'.format(i, i))
                f.write('out.append(dist_{})\n'.format(i))

    def fit_pattern_indexer(self, exercise_patterns, occur_threshold=3, is_sqlite_solver=False, plot_histogram=False):
        def write_list_to_file(myList, output_path, var_name):
            with open(output_path, 'w') as f:
                f.write("{}='''".format(var_name))
                f.write('\n'.join([str(x) for x in myList]))
                f.write("'''.split('\\n')")
        
        if is_sqlite_solver:
            conn = sqlite3.connect('temporary.db')
            cursor = conn.cursor()

            # Create table
            cursor.execute('''CREATE TABLE patterns (pattern_string text primary key, occurrence int)''')

            with tqdm(total=len(exercise_patterns)) as pbar:
                for ptn in exercise_patterns:
                    input_path = ptn['file_path']
                    pbar.set_description("Processing file {}".format(input_path))

                    try:
                        with open(input_path, 'r') as f:
                            content = filter(lambda line: '#SUP: 1\t' not in line, f.readlines())
                    except IOError:
                        print('No such file or directory: "{}"'.format(input_path))
                        pbar.update(1)
                        continue

                    patterns = [x[:x.find('#')].strip() for x in content]            
                    for p in patterns: 
                        row = cursor.execute('SELECT * FROM patterns WHERE pattern_string="{}"'.format(p)).fetchone()
                        if row is None:
                            cursor.execute("INSERT INTO patterns VALUES(?, ?)", (p, 1))
                        else:
                            cursor.execute("UPDATE patterns SET occurrence=? WHERE pattern_string=?", ((int(row[1]) + 1), p))
                    conn.commit()

                    f.close()
                    pbar.update(1)

            data = cursor.execute('SELECT pattern_string FROM patterns WHERE occurrence>={}'.format(occur_threshold)).fetchall()
            patterns_keep = [x[0] for x in data]
            n = cursor.execute('SELECT count(*) FROM patterns'.format(occur_threshold)).fetchall()[0][0]
            print('{} unique patterns are found'.format(n))
            print('{} patterns are removed from the white list'.format(n - len(patterns_keep)))    
            
            # We can also close the connection if we are done with it.
            # Just be sure any changes have been committed or they will be lost.
            conn.close()
            os.remove('temporary.db')
        else:
            result = {}
            with tqdm(total=len(exercise_patterns)) as pbar:
                for ptn in exercise_patterns:
                    input_path = ptn['file_path']
                    pbar.set_description("Processing file {}".format(input_path))
                    
                    try:
                        f = open(input_path, 'r')
                    except IOError:
                        print('No such file or directory: "{}"'.format(input_path))
                        pbar.update(1)
                        continue

                    for line in f:
                        if '#SUP: 1\t' in line:
                            continue
                        p = line[:line.find('#')].strip()   
                        if p not in result:
                            result[p] = 1
                        else:
                            result[p] += 1  
                    
                    f.close()
                    pbar.update(1)

            patterns_keep = [k for k, v in result.items() if v >= occur_threshold]
            print('{} unique patterns are found'.format(len(result)))
            print('{} patterns are removed from the white list'.format(len(result) - len(patterns_keep)))    

        self.pattern_idxer = ItemSetIndexer()
        self.pattern_idxer.fit(patterns_keep)

    def _count_patterns(self, patterns, n=5):
        user_count = len(patterns)
        #patterns = [eval(x) for x in patterns]
        patterns_flattened = reduce(lambda a,b: a+b, patterns)
        patterns_set = list(set(patterns_flattened))    
        patterns_decoded = list(self.pattern_idxer.decode(patterns_set))
        
        dic = {}
        for p in patterns_flattened:
            colname = patterns_decoded[patterns_set.index(p)]
            if colname not in dic:
                dic[colname] = 0
            dic[colname] += 1
        
        data = [[k, dic[k]] for k in sorted(dic, key=dic.get, reverse=True)]
        data = [[x[0], "{:.2f}".format(100. * x[1] / user_count)] for x in data[:n]]
        
        pattern_strings = [
            " & ".join(["{}=({})".format(a,b) for a,b in x])
            for x in self.record_idxer.decode([x[0].split() for x in data])
        ]
        data = [[a, b[1]] for a,b in zip(pattern_strings, data)]    
        
        groups = {}
        for row in data:
            matches = re.findall(r"exerciseClassId=\(([-+]?\d*\.\d+|\d+)", row[0])
            matches += re.findall(r"exerciseClassId=([-+]?\d*\.\d+|\d+)", row[0])
            matches += re.findall(r"isExercise=\(0\)", row[0])        
            matches = list(set(matches))
            for x in matches:
                if x not in groups:
                    groups[x] = [row]
                else:
                    groups[x] += [row]
                    
        if len(matches) == 0:
            matches = list(set(re.findall(r"isExercise=\(1\)", row[0])))
            for x in matches:
                if x not in groups:
                    groups[x] = [row]
                else:
                    groups[x] += [row]
        
        for k in groups:
            itemset = groups[k]
            df = pd.DataFrame(itemset, columns=['pattern', 'percentage'])
            display(df)
        
        if len(groups.keys()) == 1 and list(groups.keys())[0] == 'isExercise=(0)':
            df = pd.DataFrame(data, columns=['pattern', 'percentage'])
            display(df.head())

    def _run_PFPM(self, input_path, file_path, minper=0, maxper=10, minavgper=0, maxavgper=7):
        subprocess.call([
            'java', '-jar', 'utils/spmf.jar', 'run', 'PFPM', 
            input_path, file_path, str(minper), str(maxper), str(minavgper), str(maxavgper)
        ])

    def _run_FPM(self, input_path, file_path, support=0.1):
        subprocess.call([
            'java', '-jar', 'utils/spmf.jar', 'run', 'FPGrowth_itemsets', 
            input_path, file_path, '{}%'.format(support*100)
        ])
    
    def decode_exercise_patterns(self, exercise_patterns, output_path='PFPM/result/'):
        exercise_patterns_decoded = []
        with tqdm(total=len(exercise_patterns)) as pbar:
            for ptn in exercise_patterns:
                input_path = ptn['file_path']
                userID, time_window, _ = ptn['file_path'].split('/')[-1].split('.')
                pbar.set_description("Processing file {} ".format(input_path))

                file_path = os.path.join(output_path, '{}.{}.txt'.format(userID, time_window))
                exercise_patterns_decoded.append({
                    'user': userID,
                    'time_window': time_window,
                    'file_path': file_path
                })
                self._process_output(input_path, file_path)
        return exercise_patterns_decoded

    def _pattern_decoding(self, p):
        Y, info = p[:p.find('#')].split(), p[p.find('#')+1:].split(' #')
        [X] = self.record_idxer.decode([Y])
        trx = ' | '.join(['{}=({})'.format(a,b) for a,b in X])
        return trx, info  
    
    def _process_output(self, output_path, result_path):  
        with open(output_path, 'r') as f:
            content = f.readlines()
        patterns = [x.strip() for x in content]
        
        if len(patterns) > 0:        
            rows = [self._pattern_decoding(p) for p in patterns]
            content = '\n'.join(
                ['{}\t{}'.format(trx, '\t'.join(info)) for (trx, info) in rows]
            )
            
            with open(result_path, 'w') as f:
                f.write(content)
        return patterns
    
    def _dateMatching(self, df, date_column='exerciseDate'):
        date_list = list(pd.date_range(df[date_column].min().date(), 
                                       df[date_column].max().date()))
        template = pd.DataFrame(list(pd.date_range(df[date_column].min().date(), 
                                                   df[date_column].max().date())), 
                                columns=[date_column])

        df = df.merge(template, on=[date_column], how='outer')
        df['isExercise'] = df['isExercise'].fillna(value=0)
        df = df.sort_values(by=date_column)    
        return df

    def _get_time_window_list(self, df, date_column='exerciseDate'):
        _min, _max = df[date_column].min().date(), df[date_column].max().date()
        return list(pd.date_range('{}-{}-01'.format(_min.year, _min.month), 
                                  '{}-{}-01'.format(_max.year, _max.month), freq='M'))

    def _fetch_sequence_in_window(self, df, dt, date_column='exerciseDate'):
        data = df.loc[
            (df[date_column] >= datetime(dt.year, dt.month, 1)) & \
            (df[date_column] < datetime(dt.year, dt.month, 1) + relativedelta(months=1))
        ]
        return data

    def _write_file(self, data, file_path, date_column='exerciseDate'):
        def write_trx(f, trx):
            trx = list(set(trx))
            trx.sort()
            f.write(' '.join(trx) + '\n')

        with open(file_path, 'w') as f:
            X = data[self.itemList].values.tolist()
            D = data[[date_column]].values.tolist()
            
            Y = [a+b for (a,b) in zip(
                data[[date_column]].values.tolist(), self.record_idxer.encode(X)
            )]
            
            d0, trx = None, []
            for row in Y:
                d1, items = row[0], row[1:]     
                
                if d0 and d1 != d0:
                    write_trx(f, trx)
                    trx = []
                
                trx += items
                d0 = d1
                
            write_trx(f, trx)
        
    def _create_combination(self, df, combinationList):
        def create_value(valueForm, x):
            x = list(x)
            if 'nan' in [str(i) for i in x]:
                return np.NaN
            else:
                if len(x) == 2:
                    return valueForm.format(x[0], x[1])
                else:
                    return valueForm.format(x[0], x[1], x[2])

        valueForm = ' AND '.join(['%s={}' % col for col in combinationList])
        return [create_value(valueForm, x) for x in df[combinationList].itertuples(index=False, name=None)]
    
    # def chain_list_alpha(self, normalizer, corpus_sents, progress_bar = False):
    #     # TODO: maybe use the self.normalizer

    #     chain_list = []
    #     alphabet = Counter()
        
    #     if progress_bar:
    #         for sent in tqdm(corpus_sents, desc="Normalize & Alphabet"):
    #             sent = normalizer.normalize(sent)
    #             sent = list(sent)
    #             if sent !=[]:
    #                 chain_list += sent
    #                 alphabet.update(Counter(sent))
    #     else:
    #         for sent in corpus_sents:
    #             sent = normalizer.normalize(sent)
    #             sent = list(sent)
    #             if sent !=[]:
    #                 chain_list += sent
    #                 alphabet.update(Counter(sent))
                    
    #     return chain_list, alphabet
    
    # def find_best_pair(self, chain_list, progress_bar = False):
        
    #     pair_count = Counter()
    #     if progress_bar:
    #         for pair in tqdm(list(zip(chain_list, chain_list[1:])), desc="Finding Best Pair"):
    #             pair_count[pair] += 1            
        
    #     else:
    #         for pair in list(zip(chain_list, chain_list[1:])):
    #             pair_count[pair] += 1            
        
    #     return pair_count
        
    
    # def findall_contexts(self,chain,best_pair_string,re_voc_l,re_voc_r):
    #     contexts = re.findall(re_voc_l+best_pair_string+re_voc_r, chain, overlapped=True)
    #     return contexts
            
#     def build_new(
#         self,
#         corpus = None,
#         vocab_size = None,
#         special_tokens = None,
#         save = False,
#         save_step = None,
#         progress_bar = True,
#         resume_merges = False,
#         parallel = True,
#         sparse = True,
#         sparse_mode = "csr",
#         corpus_length = None
#         ):
#         """
#         Build vocabulary from a Corpus.
#         Vocabularies can be extended by providing an existing merging list. If resume_merges = True, the current merges in self.merges will be used. Otherwise one can provide a list of merges as value of resume_merges.
        
#         If vocab_size is negative, the value is taken, not as the target size of the voc, but as the number of new terms to compute beyond the size of the initial alphabet
#         """
        
#         if corpus == None:
#             corpus = self.name
        
#         if vocab_size == None:
#             vocab_size = self.config.size
        
#         if special_tokens == None:
#             special_tokens = self.config.special_tokens
        
#         if save == True and save_step != None:
#             saveQ = True
            
#             if not isdir(self.path):
#                 makedirs(self.path)
                
#             # save_steps = {save_step*i for i in range(int(abs(vocab_size)/save_step)+1)}
#         else:
#             saveQ = False

#         def parallel_chain(chain, n_of_parts, overlap = 0):
#             """
#             Breaks the chain in n chunks to compute best pair of terms. Chunks are overlapping by one term, so as no pair of terms is lost due to the break.
#             """
#             if not isinstance(chain,list):
#                 chain = list(chain)
#             chunk_size = int(len(chain) / n_of_parts)+1
#             for i in range(0, len(chain), chunk_size):
#                 yield chain[i : i + chunk_size + overlap]
                
#         def extract_drc(pairs, encoder: dict):
#             data = []
#             rows = []
#             columns = []
#             for (r,c),d in pairs:
#                 data.append(d)
#                 rows.append(encoder[r])
#                 columns.append(encoder[c])
#             return data, rows, columns

#         def separate_chain(chain, n_of_parts, best_pair: list):
#             """
#             Separate a chain (in list form) for parallel processing of regex findall of pair, taking care that the cuts of the chunks don't fall in the neiborhood of the pair, affecting the final counts
#             """
#             chunk_size = int(len(chain) / n_of_parts)+1
#             b = 0
#             n = chunk_size
#             chain_len = len(chain)
#             for i in range(n_of_parts):
#                 n = (i+1)*chunk_size
#                 if chain_len > n:
#                     while chain[n-2:n] == best_pair or chain[n-1:n+1] == best_pair:
#                         n = n+1
#                 yield ("[SEP_i] " if i!=0 else "") + " ".join(chain[b:n]) + (" [SEP_i]" if i!=n_of_parts-1 else "")
#                 b = n-1

#         def agglutinate_chain(pair, cl_chain):
#             bigram = re.escape(" ".join(pair))
#             p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
#             cl_chain = p.sub("".join(pair), cl_chain)
#             return cl_chain
        
#         # TODO: maybe use the self.normalizer
#         if isinstance(self.config.normalizer,list):
#             normalizer = eval(
#                 f"tokenizer.normalizers.Sequence({self.config.normalizer})"
#                 )
#         else:
#             normalizer = eval(
#                 f"tokenizer.normalizers.{util.if_none_disable(self.config.normalizer)}"
#             )
        
#         if parallel:
            
#             par_corpus = parallel_chain(self.corpus.train[:corpus_length], self.cpu_count)

#             print("Normalize & Alphabet in parallel...")
#             # TODO: maybe use the self.normalizer
#             start = datetime.now()
#             result = util.multiprocessing(partial(self.chain_list_alpha, normalizer), par_corpus, cores=self.cpu_count) #, desc="Normalize & Alphabet")
            
#             chain_list = []
#             alphabet = Counter()
#             for chain_l, alpha in result:
#                 chain_list += chain_l
#                 alphabet += alpha
#             print(f"Normalize & Alphabet computed in {datetime.now()-start}")
                
#         else:
#             chain_list, alphabet = self.chain_list_alpha(normalizer, self.corpus.train[:corpus_length], progress_bar=True)
        

#         # TODO: Add resume feature        
#         if resume_merges != False and False:
#             if resume_merges == True:
#                 merges = self.merges
#             elif isinstance(resume_merges,list):
#                 merges = resume_merges
            
#             for pair in tqdm(merges, desc = "Resuming Existing Vocabulary",disable = not progress_bar):
#                 chain_list = agglutinate_chain(tuple(pair.split()),chain_list)
#             vocabulary = Counter()
#             for term in tqdm(chain_list,desc="Building Resumed Vocabulary", disable = not progress_bar):
#                 vocabulary[term] += 1
            
#         else:
#             merges = []
#             vocabulary = alphabet

#         if parallel:
#             print("Computing matrix values in parallel...")
#             start = datetime.now()
            
#             par_chain = parallel_chain(chain_list, self.cpu_count, overlap=1)
            
#             result = util.multiprocessing(self.find_best_pair, par_chain, cores=self.cpu_count) 
                                
#             pairs = reduce(operator.add, result)
#             pairs = pairs.most_common()
#             print(f"Matrix values computed in {datetime.now()-start}")
            
#         else:
#             pairs = self.find_best_pair(chain_list, progress_bar = True)
#             pairs = pairs.most_common()


#         cl_chain = "[SEP] "+" ".join(chain_list)+" [SEP]"
#         encode = {k:i for i,(k,v) in enumerate(alphabet.most_common())}
#         decode = {i:k for k,i in encode.items()}
        
#         voc_len = len(encode)
#         special_tokens_len = 0 if special_tokens == None else len(special_tokens)
        
#         print(f"Alphabet Size: {voc_len}")
        
#         if vocab_size<0:
#             voc_final_length = voc_len + abs(vocab_size)
#         else:
#             voc_final_length = vocab_size - special_tokens_len
        
#         # TODO: this can yield an error if vocab_size is smaller than alphabet. Fix to make vocab_size = max(vocab_size, len(alphabet)), and don't launch matrix construction in that case.
#         if sparse:
#             data, rows, columns = extract_drc(pairs,encode)
#             voc_matrix = coo_matrix((np.array(data), (np.array(rows),np.array(columns))), shape=(voc_final_length, voc_final_length), dtype=int)

#         else:
#             voc_matrix = np.zeros((voc_final_length, voc_final_length), dtype=int)
#             for (row,column),value in pairs:
#                 voc_matrix[encode[row], encode[column]] = value
        
#         delta_voc = voc_final_length - voc_len
#         best_pair = "init"
#         pair_count = "---"
#         new_i = voc_len

#         # TODO: remove SparseEfficiencyWarning

#         t = trange(delta_voc, disable = not progress_bar)


#         profile = Profiler()
#         profile.start()
        
#         for i in t:
#             t.set_description(f"Pair: {best_pair}, {pair_count}")
#             t.refresh()

#             if sparse:
#                 max_i = voc_matrix.data.argmax()
#                 pair_row = voc_matrix.row[max_i]
#                 pair_col = voc_matrix.col[max_i]
#                 pair_count = voc_matrix.data[max_i]
#             else:
#                 pair_row,pair_col = np.unravel_index(np.argmax(voc_matrix, axis=None), voc_matrix.shape)
#                 pair_count = voc_matrix[pair_row,pair_col]
            
#             if pair_count == 0:
#                 break

#             best_pair = (decode[pair_row], decode[pair_col])
#             best_pair_string = " ".join(best_pair)
#             merges.append(best_pair_string)
#             best_pair_string_voc = "".join(best_pair)
#             re_voc_l = "("+"|".join([" "+k+" " for k in encode.keys()]+["\[SEP\] ","\[SEP_i\] "])+")"
#             re_voc_r = "("+"|".join([" "+k+" " for k in encode.keys()]+[" \[SEP\]"," \[SEP_i\]"])+")"
#             if parallel:
#                 chain_chunks = list(
#                     separate_chain(cl_chain.split(), self.cpu_count, list(best_pair)))
#                 result = util.multiprocessing(
#                     partial(self.findall_contexts,best_pair_string=best_pair_string,re_voc_l=re_voc_l,re_voc_r=re_voc_r),chain_chunks,
#                     cores = self.cpu_count
#                     )
                
#                 merge_context = reduce(operator.add, result)
#             else:
#                 merge_context = re.findall(re_voc_l+best_pair_string+re_voc_r, cl_chain, overlapped=True)
#             merge_context_count_l = Counter()
#             merge_context_count_r = Counter()
#             for l,r in merge_context:
#                 if "[SEP]" not in l:
#                     merge_context_count_l[encode[l.strip()]] += 1
#                 if "[SEP]" not in r:
#                     merge_context_count_r[encode[r.strip()]] += 1
            
#             if sparse:
#                 # Convert matrix to CSR or LIL, for item attribution and arithmetic 
#                 if sparse_mode == "csr":
#                     voc_matrix = voc_matrix.tocsr()
#                 else:
#                     voc_matrix = voc_matrix.tolil()
            
#             for row,key in merge_context_count_l.items():
#                 voc_matrix[row,new_i] = key
                
#             for column,key in merge_context_count_r.items():
#                 voc_matrix[new_i,column] = key

#             # Correct previous counts
            
#             # compute #(l,r)-(l,r)
#             pair_pair_count = len(re.findall(" "+best_pair_string+" "+best_pair_string+" ", cl_chain, overlapped=False))
#             # remove #(l,r)-(l,r) from (l,r)-l
#             voc_matrix[new_i,pair_row] -= pair_pair_count
#             # remove #(l,r)-(l,r) from r-(l,r)
#             voc_matrix[pair_col,new_i] -= pair_pair_count
#             # remove #(l,r)-(l,r) from r-l
#             voc_matrix[pair_col,pair_row] -= pair_pair_count
#             # substract (l,r)- from r-
#             voc_matrix[pair_col,:new_i] -= voc_matrix[new_i,:new_i]
#             # substract -(l,r)- from -l
#             voc_matrix[:new_i,pair_row] -= voc_matrix[:new_i,new_i]
            
#             # set l-r to 0
#             voc_matrix[pair_row,pair_col] = 0
#             # register #(l,r)-(l,r)
#             voc_matrix[new_i,new_i] = pair_pair_count
            
#             if sparse:
#                 # Convert matrix back to COO, to restart the loop
#                 voc_matrix = voc_matrix.tocoo()
            
#             best_pair_string_voc = "".join(best_pair)
#             encode[best_pair_string_voc] = new_i
#             decode[new_i] = best_pair_string_voc
#             new_i += 1
#             cl_chain = agglutinate_chain(best_pair_string.split(),cl_chain)

#             if saveQ == True:
#                 voc_partial_len = voc_len + special_tokens_len + i + 1
#                 if voc_partial_len % save_step == 0:
                    
#                     if sparse:
#                         freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
#                     else:
#                         freq_values = voc_matrix.sum(axis=1).T.tolist()
#                     vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
#                     vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
                    
#                     if special_tokens != None:
#                         vocabulary = vocabulary + [(token,0) for token in special_tokens]

#                     self.merges = merges
#                     self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#                     self.freq = dict(vocabulary)
#                     self.alpha = dict(alphabet)
#                     step_path = self.path / str(voc_partial_len)
#                     self.save(step_path)
#                     print(f"Intermediate vocabulary saved to {step_path}")

#         profile.stop()
#         print(profile.output_text(unicode=True, color=True))

#         if sparse:
#             freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
#         else:
#             freq_values = voc_matrix.sum(axis=1).T.tolist()
#         vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
#         vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
        
#         if special_tokens != None:
#             vocabulary = vocabulary + [(token,0) for token in special_tokens]
        
#         self.merges = merges
#         self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#         self.freq = dict(vocabulary)
#         self.alpha = dict(alphabet.most_common())

#         self.decode = {i:k for k,i in self.encode.items()}
        
#         self.len = len(vocabulary)     
#         self.freq_mass = sum(self.freq.values())
#         self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

#         print("Vocabulary built")
        
#         if save == True:
#             self.save()
#             print(f"Vocabulary saved to {self.path}")
    
    
#     def build_old(
#     self,
#     corpus = None,
#     vocab_size = None,
#     special_tokens = None,
#     save = False,
#     save_step = None,
#     progress_bar = True,
#     resume_merges = False,
#     parallel = False,
#     parallel_mode = "process",
#     corpus_length = None
#     ):
#         """
#         Build vocabulary from a Corpus.
#         Vocabularies can be extended by providing an existing merging list. If resume_merges = True, the current merges in self.merges will be used. Otherwise one can provide a list of merges as value of resume_merges.
        
#         If vocab_size is negative, the value is taken, not as the target size of the voc, but as the number of new terms to compute beyond the size of the initial alphabet
#         """
        
#         if corpus == None:
#             corpus = self.name
        
#         if vocab_size == None:
#             vocab_size = self.config.size
        
#         if special_tokens == None:
#             special_tokens = self.config.special_tokens
        
#         if save == True and save_step != None:
#             saveQ = True
            
#             if not isdir(self.path):
#                 makedirs(self.path)
                
#         else:
#             saveQ = False
        
#         def parallel_chain(chain, n_of_parts, overlap = 0):
#             """
#             Breaks the chain in n chunks to compute best pair of terms. Chunks are overlapping by one term, so as no pair of terms is lost due to the break.
#             """
#             if not isinstance(chain,list):
#                 chain = list(chain)
#             chunk_size = int(len(chain) / n_of_parts)+1
#             for i in range(0, len(chain), chunk_size):
#                 yield chain[i : i + chunk_size + overlap]

        
#         def agglutinate_chain(pair, chain_list):
#             chain_list = " ".join(chain_list) 
#             bigram = re.escape(" ".join(pair))
#             p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
#             chain_list = p.sub("".join(pair), chain_list)
#             chain_list = chain_list.split()
#             return chain_list
        
#         # TODO: maybe use the self.normalizer
#         if isinstance(self.config.normalizer,list):
#             normalizer = eval(
#                 f"tokenizer.normalizers.Sequence({self.config.normalizer})"
#                 )
#         else:
#             normalizer = eval(
#                 f"tokenizer.normalizers.{util.if_none_disable(self.config.normalizer)}"
#             )
        
        
#         if parallel:
            
#             par_corpus = parallel_chain(self.corpus.train[:corpus_length], self.cpu_count)
            
#             # TODO: maybe use the self.normalizer
#             if parallel_mode == "process":
#                 result = util.multiprocessing(partial(self.chain_list_alpha, normalizer), par_corpus, cores=self.cpu_count) # , desc="Normalize & Alphabet")               
#             else:
#                 result = util.multithreading(partial(self.chain_list_alpha, normalizer), par_corpus, cores=self.cpu_count)
            
#             chain_list = []
#             alphabet = Counter()
#             for chain_l, alpha in result:
#                 chain_list += chain_l
#                 alphabet += alpha
                
#         else:
#             # TODO: maybe use the self.normalizer
#             chain_list, alphabet = self.chain_list_alpha(normalizer, self.corpus.train[:corpus_length], progress_bar=True)

        
#         if resume_merges != False:
#             if resume_merges == True:
#                 merges = self.merges
#             elif isinstance(resume_merges,list):
#                 merges = resume_merges
            
#             for pair in tqdm(merges, desc = "Resuming Existing Vocabulary",disable = not progress_bar):
#                 chain_list = agglutinate_chain(tuple(pair.split()),chain_list)
#             vocabulary = Counter()
#             for term in tqdm(chain_list,desc="Building Resumed Vocabulary", disable = not progress_bar):
#                 vocabulary[term] += 1
            
#         else:
#             merges = []
#             vocabulary = alphabet
        
        
#         special_tokens_len = 0 if special_tokens == None else len(special_tokens)
#         voc_len = len(vocabulary) + special_tokens_len
#         pair = ("[init]","[init]")
        
#         print(f"Alphabet Size: {voc_len}")
        
#         if vocab_size<0:
#             delta_voc = abs(vocab_size)
#         else:
#             delta_voc = vocab_size - voc_len
        
        
#         t = trange(delta_voc, disable = not progress_bar)
        
#         for i in t:
#             t.set_description(f"Pair: {pair[0]}, {pair[1]}")
#             t.refresh()
            
#             if parallel:
                
#                 par_chain = parallel_chain(chain_list, self.cpu_count, overlap=1)
                
#                 if parallel_mode == "process":
#                     result = util.multiprocessing(self.find_best_pair, par_chain, cores=self.cpu_count)               
#                 else:
#                     result = util.multithreading(self.find_best_pair, par_chain, cores=self.cpu_count)      
                                    
#                 best_pairs = reduce(operator.add, result)
                
#             else:
#                 best_pairs = self.find_best_pair(chain_list)
                
#             pair = best_pairs.most_common(1)[0]
            
#             chain_list = agglutinate_chain(pair[0], chain_list)
            
#             merges.append(" ".join(pair[0]))
            
#             if saveQ == True:
#                 voc_partial_len = voc_len + i + 1
#                 if voc_partial_len % save_step == 0:
#                     vocabulary = Counter(chain_list).most_common()
#                     if special_tokens != None:
#                         vocabulary = vocabulary + [(token,0) for token in special_tokens]

#                     self.merges = merges
#                     self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#                     self.freq = dict(vocabulary)
#                     self.alpha = dict(alphabet)
#                     step_path = self.path / str(voc_partial_len)
#                     self.save(step_path)
#                     print(f"Intermediate vocabulary saved to {step_path}")
        
#         vocabulary = Counter()            
#         for term in tqdm(chain_list, desc="Building Final Vocabulary", disable = not progress_bar):
#             vocabulary[term] += 1
#         vocabulary = vocabulary.most_common()
        
        
#         if special_tokens != None:
#             vocabulary = vocabulary + [(token,0) for token in special_tokens]

#         self.merges = merges
#         self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#         self.freq = dict(vocabulary)
#         self.alpha = dict(alphabet.most_common())

#         self.decode = {i:k for k,i in self.encode.items()}
        
#         self.len = len(vocabulary)     
#         self.freq_mass = sum(self.freq.values())
#         self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

#         print("Vocabulary built")
        
#         if save == True:
#             self.save()
#             print(f"Vocabulary saved to {self.path}")


# ###########

#     def build_par(
#         self,
#         corpus = None,
#         vocab_size = None,
#         special_tokens = None,
#         save = False,
#         save_step = None,
#         progress_bar = True,
#         resume_merges = False,
#         parallel = True,
#         sparse = True,
#         sparse_mode = "csr",
#         corpus_length = None
#         ):
#         """
#         Build vocabulary from a Corpus.
#         Vocabularies can be extended by providing an existing merging list. If resume_merges = True, the current merges in self.merges will be used. Otherwise one can provide a list of merges as value of resume_merges.
        
#         If vocab_size is negative, the value is taken, not as the target size of the voc, but as the number of new terms to compute beyond the size of the initial alphabet
#         """
        
#         if corpus == None:
#             corpus = self.name
        
#         if vocab_size == None:
#             vocab_size = self.config.size
        
#         if special_tokens == None:
#             special_tokens = self.config.special_tokens
        
#         if save == True and save_step != None:
#             saveQ = True
            
#             if not isdir(self.path):
#                 makedirs(self.path)
                
#             # save_steps = {save_step*i for i in range(int(abs(vocab_size)/save_step)+1)}
#         else:
#             saveQ = False

#         def parallel_chain(chain, n_of_parts, overlap = 0):
#             """
#             Breaks the chain in n chunks to compute best pair of terms. "overlap" gives the number of terms of overlap between chunks to alow to ensure that no pair of terms is lost due to the break.
#             """
#             chunk_size = int(len(chain) / n_of_parts)+1            
#             for i in range(0, len(chain), chunk_size):
#                 yield chain[i : i + chunk_size + overlap]
                
#         def extract_drc(pairs, encoder: dict):
#             data = []
#             rows = []
#             columns = []
#             for (r,c),d in pairs:
#                 data.append(d)
#                 rows.append(encoder[r])
#                 columns.append(encoder[c])
#             return data, rows, columns

#         def separate_chain(chain, n_of_parts, best_pair: list):
#             """
#             Separate a chain (in list form) for parallel processing of regex findall of pair, taking care that the cuts of the chunks don't fall in the neighborhood of the pair, affecting the final counts
#             """
#             chunk_size = int(len(chain) / n_of_parts)+1
#             b = 0
#             n = chunk_size
#             chain_len = len(chain)
#             for i in range(n_of_parts):
#                 n = (i+1)*chunk_size
#                 if chain_len > n:
#                     while chain[n-2:n] == best_pair or chain[n-1:n+1] == best_pair:
#                         n = n+1
#                 yield ("[SEP_i] " if i!=0 else "") + " ".join(chain[b:n]) + (" [SEP_i]" if i!=n_of_parts-1 else "")
#                 b = n-1

#         def agglutinate_chain(pair, cl_chain):
#             bigram = re.escape(" ".join(pair))
#             p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
#             cl_chain = p.sub("".join(pair), cl_chain)
#             return cl_chain
        
#         # if isinstance(self.config.normalizer,list):
#         #     normalizer = eval(
#         #         f"tokenizer.normalizers.Sequence({self.config.normalizer})"
#         #         )
#         # else:
#         #     normalizer = eval(
#         #         f"tokenizer.normalizers.{util.if_none_disable(self.config.normalizer)}"
#         #     )
        
#         if parallel:
            
#             with Parallel(n_jobs=self.cpu_count) as parallel_pool:
                
#                 par_corpus = parallel_chain(self.corpus.train[:corpus_length], self.cpu_count)

#                 print("Normalize & Alphabet in parallel...")
#                 start = datetime.now()
                
#                 # TODO: check if chain_list_alpha could be done without for loop
#                 result = parallel_pool(delayed(self.chain_list_alpha)(self.normalizer,corpus_chunk) for corpus_chunk in par_corpus)
                
#                 # TODO: check if the reduce method is not better
#                 chain_list = []
#                 alphabet = Counter()
#                 for chain_l, alpha in result:
#                     chain_list += chain_l
#                     alphabet += alpha
#                 print(f"Normalize & Alphabet computed in {datetime.now()-start}")
                
#                 # TODO: Add resume feature
#                 merges = []
#                 vocabulary = alphabet

#                 print("Computing matrix values in parallel...")
#                 start = datetime.now()
                
#                 par_chain = parallel_chain(chain_list, self.cpu_count, overlap=1)
                
                
#                 result = parallel_pool(delayed(self.find_best_pair)(chain_chunk) for chain_chunk in par_chain)
                                    
#                 pairs = reduce(operator.add, result)
#                 pairs = pairs.most_common()
#                 print(f"Matrix values computed in {datetime.now()-start}")

#                 cl_chain = "[SEP] "+" ".join(chain_list)+" [SEP]"
#                 encode = {k:i for i,(k,v) in enumerate(alphabet.most_common())}
#                 decode = {i:k for k,i in encode.items()}
                
#                 voc_len = len(encode)
#                 special_tokens_len = 0 if special_tokens == None else len(special_tokens)
                
#                 print(f"Alphabet Size: {voc_len}")
                
#                 if vocab_size<0:
#                     voc_final_length = voc_len + abs(vocab_size)
#                 else:
#                     voc_final_length = vocab_size - special_tokens_len
                    
#                 # TODO: this can yield an error if vocab_size is smaller than alphabet. Fix to make vocab_size = max(vocab_size, len(alphabet)), and don't launch matrix construction in that case.
#                 if sparse:
#                     data, rows, columns = extract_drc(pairs,encode)
#                     voc_matrix = coo_matrix((np.array(data), (np.array(rows),np.array(columns))), shape=(voc_final_length, voc_final_length), dtype=int)

#                 else:
#                     voc_matrix = np.zeros((voc_final_length, voc_final_length), dtype=int)
#                     for (row,column),value in pairs:
#                         voc_matrix[encode[row], encode[column]] = value
                
#                 delta_voc = voc_final_length - voc_len
#                 best_pair = "init"
#                 pair_count = "---"
#                 new_i = voc_len


#                 # TODO: remove SparseEfficiencyWarning

#                 t = trange(delta_voc, disable = not progress_bar)

                
#                 for i in t:
#                     t.set_description(f"Pair: {best_pair}, {pair_count}")
#                     t.refresh()

#                     if sparse:
#                         max_i = voc_matrix.data.argmax()
#                         pair_row = voc_matrix.row[max_i]
#                         pair_col = voc_matrix.col[max_i]
#                         pair_count = voc_matrix.data[max_i]
#                     else:
#                         pair_row,pair_col = np.unravel_index(np.argmax(voc_matrix, axis=None), voc_matrix.shape)
#                         pair_count = voc_matrix[pair_row,pair_col]
                    
#                     if pair_count == 0:
#                         break

#                     best_pair = (decode[pair_row], decode[pair_col])
#                     best_pair_string = " ".join(best_pair)
#                     merges.append(best_pair_string)
#                     best_pair_string_voc = "".join(best_pair)
#                     re_voc_l = "("+"|".join([" "+k+" " for k in encode.keys()]+["\[SEP\] ","\[SEP_i\] "])+")"
#                     re_voc_r = "("+"|".join([" "+k+" " for k in encode.keys()]+[" \[SEP\]"," \[SEP_i\]"])+")"


#                     chain_chunks = list(
#                         separate_chain(cl_chain.split(), self.cpu_count, list(best_pair)))
#                     result = parallel_pool(delayed(self.findall_contexts)(chain_chunk,best_pair_string,re_voc_l,re_voc_r) for chain_chunk in chain_chunks)
#                     merge_context = reduce(operator.add, result)
                            
#                     l_context = [l for l,r in merge_context]
#                     merge_context_count_l = Counter(l_context)
#                     merge_context_count_l = {encode[k.strip()]:v for k,v in merge_context_count_l.most_common() if "[SEP]" not in k}
                    
#                     r_context = [r for l,r in merge_context]
#                     merge_context_count_r = Counter(r_context)
#                     merge_context_count_r = {encode[k.strip()]:v for k,v in merge_context_count_r.most_common() if "[SEP]" not in k}
                    
#                     if sparse:
#                         # Convert matrix to CSR or LIL, for item attribution and arithmetic 
#                         if sparse_mode == "csr":
#                             voc_matrix = voc_matrix.tocsr()
#                         else:
#                             voc_matrix = voc_matrix.tolil()
                    
#                     for row,key in merge_context_count_l.items():
#                         voc_matrix[row,new_i] = key
                        
#                     for column,key in merge_context_count_r.items():
#                         voc_matrix[new_i,column] = key

#                     # Correct previous counts
                    
#                     # compute #(l,r)-(l,r)
#                     pair_pair_count = len(re.findall(" "+best_pair_string+" "+best_pair_string+" ", cl_chain, overlapped=False))
#                     # remove #(l,r)-(l,r) from (l,r)-l
#                     voc_matrix[new_i,pair_row] -= pair_pair_count
#                     # remove #(l,r)-(l,r) from r-(l,r)
#                     voc_matrix[pair_col,new_i] -= pair_pair_count
#                     # remove #(l,r)-(l,r) from r-l
#                     voc_matrix[pair_col,pair_row] -= pair_pair_count
#                     # substract (l,r)- from r-
#                     voc_matrix[pair_col,:new_i] -= voc_matrix[new_i,:new_i]
#                     # substract -(l,r)- from -l
#                     voc_matrix[:new_i,pair_row] -= voc_matrix[:new_i,new_i]
                    
#                     # set l-r to 0
#                     voc_matrix[pair_row,pair_col] = 0
#                     # register #(l,r)-(l,r)
#                     voc_matrix[new_i,new_i] = pair_pair_count
                    
#                     if sparse:
#                         # Convert matrix back to COO, to restart the loop
#                         voc_matrix = voc_matrix.tocoo()
                    
#                     best_pair_string_voc = "".join(best_pair)
#                     encode[best_pair_string_voc] = new_i
#                     decode[new_i] = best_pair_string_voc
#                     new_i += 1
#                     cl_chain = agglutinate_chain(best_pair_string.split(),cl_chain)

#                     if saveQ == True:
#                         voc_partial_len = voc_len + special_tokens_len + i + 1
#                         if voc_partial_len % save_step == 0:
                            
#                             if sparse:
#                                 freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
#                             else:
#                                 freq_values = voc_matrix.sum(axis=1).T.tolist()
#                             vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
#                             vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
                            
#                             if special_tokens != None:
#                                 vocabulary = vocabulary + [(token,0) for token in special_tokens]

#                             self.merges = merges
#                             self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#                             self.freq = dict(vocabulary)
#                             self.alpha = dict(alphabet)
#                             step_path = self.path / str(voc_partial_len)
#                             self.save(step_path)
#                             print(f"Intermediate vocabulary saved to {step_path}")
                            
                            
#         else:
#             chain_list, alphabet = self.chain_list_alpha(self.normalizer, self.corpus.train[:corpus_length], progress_bar=True)
        
#             merges = []
#             vocabulary = alphabet

#             pairs = self.find_best_pair(chain_list, progress_bar = True)
#             pairs = pairs.most_common()


#             cl_chain = "[SEP] "+" ".join(chain_list)+" [SEP]"
#             encode = {k:i for i,(k,v) in enumerate(alphabet.most_common())}
#             decode = {i:k for k,i in encode.items()}
            
#             voc_len = len(encode)
#             special_tokens_len = 0 if special_tokens == None else len(special_tokens)
            
#             print(f"Alphabet Size: {voc_len}")
            
#             if vocab_size<0:
#                 voc_final_length = voc_len + abs(vocab_size)
#             else:
#                 voc_final_length = vocab_size - special_tokens_len
        
#             # TODO: this can yield an error if vocab_size is smaller than alphabet. Fix to make vocab_size = max(vocab_size, len(alphabet)), and don't launch matrix construction in that case.
#             if sparse:
#                 data, rows, columns = extract_drc(pairs,encode)
#                 voc_matrix = coo_matrix((np.array(data), (np.array(rows),np.array(columns))), shape=(voc_final_length, voc_final_length), dtype=int)

#             else:
#                 voc_matrix = np.zeros((voc_final_length, voc_final_length), dtype=int)
#                 for (row,column),value in pairs:
#                     voc_matrix[encode[row], encode[column]] = value
            
#             delta_voc = voc_final_length - voc_len
#             best_pair = "init"
#             pair_count = "---"
#             new_i = voc_len

#             # TODO: remove SparseEfficiencyWarning

#             t = trange(delta_voc, disable = not progress_bar)


#             profile = Profiler()
#             profile.start()
            
#             for i in t:
#                 t.set_description(f"Pair: {best_pair}, {pair_count}")
#                 t.refresh()

#                 if sparse:
#                     max_i = voc_matrix.data.argmax()
#                     pair_row = voc_matrix.row[max_i]
#                     pair_col = voc_matrix.col[max_i]
#                     pair_count = voc_matrix.data[max_i]
#                 else:
#                     pair_row,pair_col = np.unravel_index(np.argmax(voc_matrix, axis=None), voc_matrix.shape)
#                     pair_count = voc_matrix[pair_row,pair_col]
                
#                 if pair_count == 0:
#                     break

#                 best_pair = (decode[pair_row], decode[pair_col])
#                 best_pair_string = " ".join(best_pair)
#                 merges.append(best_pair_string)
#                 best_pair_string_voc = "".join(best_pair)
#                 re_voc_l = "("+"|".join([" "+k+" " for k in encode.keys()]+["\[SEP\] ","\[SEP_i\] "])+")"
#                 re_voc_r = "("+"|".join([" "+k+" " for k in encode.keys()]+[" \[SEP\]"," \[SEP_i\]"])+")"
#                 if parallel:
#                     chain_chunks = list(
#                         separate_chain(cl_chain.split(), self.cpu_count, list(best_pair)))
#                     result = util.multiprocessing(
#                         partial(self.findall_contexts,best_pair_string=best_pair_string,re_voc_l=re_voc_l,re_voc_r=re_voc_r),chain_chunks,
#                         cores = self.cpu_count
#                         )
                    
#                     merge_context = reduce(operator.add, result)
#                 else:
#                     merge_context = re.findall(re_voc_l+best_pair_string+re_voc_r, cl_chain, overlapped=True)
#                 merge_context_count_l = Counter()
#                 merge_context_count_r = Counter()
#                 for l,r in merge_context:
#                     if "[SEP]" not in l:
#                         merge_context_count_l[encode[l.strip()]] += 1
#                     if "[SEP]" not in r:
#                         merge_context_count_r[encode[r.strip()]] += 1
                
#                 if sparse:
#                     # Convert matrix to CSR or LIL, for item attribution and arithmetic 
#                     if sparse_mode == "csr":
#                         voc_matrix = voc_matrix.tocsr()
#                     else:
#                         voc_matrix = voc_matrix.tolil()
                
#                 for row,key in merge_context_count_l.items():
#                     voc_matrix[row,new_i] = key
                    
#                 for column,key in merge_context_count_r.items():
#                     voc_matrix[new_i,column] = key

#                 # Correct previous counts
                
#                 # compute #(l,r)-(l,r)
#                 pair_pair_count = len(re.findall(" "+best_pair_string+" "+best_pair_string+" ", cl_chain, overlapped=False))
#                 # remove #(l,r)-(l,r) from (l,r)-l
#                 voc_matrix[new_i,pair_row] -= pair_pair_count
#                 # remove #(l,r)-(l,r) from r-(l,r)
#                 voc_matrix[pair_col,new_i] -= pair_pair_count
#                 # remove #(l,r)-(l,r) from r-l
#                 voc_matrix[pair_col,pair_row] -= pair_pair_count
#                 # substract (l,r)- from r-
#                 voc_matrix[pair_col,:new_i] -= voc_matrix[new_i,:new_i]
#                 # substract -(l,r)- from -l
#                 voc_matrix[:new_i,pair_row] -= voc_matrix[:new_i,new_i]
                
#                 # set l-r to 0
#                 voc_matrix[pair_row,pair_col] = 0
#                 # register #(l,r)-(l,r)
#                 voc_matrix[new_i,new_i] = pair_pair_count
                
#                 if sparse:
#                     # Convert matrix back to COO, to restart the loop
#                     voc_matrix = voc_matrix.tocoo()
                
#                 best_pair_string_voc = "".join(best_pair)
#                 encode[best_pair_string_voc] = new_i
#                 decode[new_i] = best_pair_string_voc
#                 new_i += 1
#                 cl_chain = agglutinate_chain(best_pair_string.split(),cl_chain)

#                 if saveQ == True:
#                     voc_partial_len = voc_len + special_tokens_len + i + 1
#                     if voc_partial_len % save_step == 0:
                        
#                         if sparse:
#                             freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
#                         else:
#                             freq_values = voc_matrix.sum(axis=1).T.tolist()
#                         vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
#                         vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
                        
#                         if special_tokens != None:
#                             vocabulary = vocabulary + [(token,0) for token in special_tokens]

#                         self.merges = merges
#                         self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#                         self.freq = dict(vocabulary)
#                         self.alpha = dict(alphabet)
#                         step_path = self.path / str(voc_partial_len)
#                         self.save(step_path)
#                         print(f"Intermediate vocabulary saved to {step_path}")

#         if sparse:
#             freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
#         else:
#             freq_values = voc_matrix.sum(axis=1).T.tolist()
#         vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
#         vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
        
#         if special_tokens != None:
#             vocabulary = vocabulary + [(token,0) for token in special_tokens]
        
#         self.merges = merges
#         self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
#         self.freq = dict(vocabulary)
#         self.alpha = dict(alphabet.most_common())

#         self.decode = {i:k for k,i in self.encode.items()}
        
#         self.len = len(vocabulary)     
#         self.freq_mass = sum(self.freq.values())
#         self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

#         print("Vocabulary built")
        
#         if save == True:
#             self.save()
#             print(f"Vocabulary saved to {self.path}")

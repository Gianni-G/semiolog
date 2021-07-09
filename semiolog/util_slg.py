# EXTERNAL MODULES



# Functions for the Chain class



#%%

# ###############################
# # 00 - GENERAL                #
# ###############################

# def execute_procedure(step_list:list):
#     for step in step_list:
#         print(f"\nExecuting step:\t{step}")
#         print(f"===============\t{'='*len(step)}\n")
#         importlib.import_module("."+step,package="analyses")
#     print("\nProcedure Finished!\n")


# ###############################
# # 01 - SEGMENTATION FUNCTIONS #
# ###############################

# # 01.01 Building Vocabulary #
# #############################

# def load_corpus(
#     filename:str,
#     length = None,
#     ):
#     print("Loading Corpus")
#     start = time.perf_counter()
#     with open(f"{paths.corpus}{filename}.txt", "r") as f:
#         chain = f.read(length)
#     finish = time.perf_counter()
#     print(f"Corpus loaded in {round(finish - start,2)} secs\n")
#     return chain

# def find_best_pair(chain_spaced):
#     pairs = Counter()
#     pre_units = chain_spaced.split()
#     for i in range(len(pre_units) - 1):
#         pairs[pre_units[i], pre_units[i + 1]] += 1
#     return pairs.most_common()[0][0]

# def agglutinate_chain(pair, chain_spaced):
#     bigram = re.escape(" ".join(pair))
#     p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
#     new_chain = p.sub("".join(pair), chain_spaced)
#     return new_chain

# def count_units_in_chain(chain_spaced):
#     vocab = Counter()
#     chain_units = chain_spaced.split()
#     for unit in chain_units:
#         vocab[unit] += 1
#     return dict(vocab.most_common())

# def build_vocab(
#     chain:str,
#     voc_length: int,
#     inter_save = [],
#     save_finalQ = True,
#     filename = "voc_default",
#     resumeQ = False,
#     corpus_name = "corp_deefault",
#     courpus_length = None,
# ):
#     if resumeQ:
#         initial_chain = chain.replace(" ", "")
#     else:
#         initial_chain = chain
#     spaced_chain = " ".join(chain)
#     alpha = count_units_in_chain(spaced_chain)
#     alpha_len = len(alpha)
#     print(f"Length of initial alphabet: {alpha_len}")
#     if resumeQ:
#         print("Resuming chain...")
#         spaced_chains = chain
#         print("Chain resumed")
#     if voc_length > 0:
#         print("Enter loop")
#         for i in range(voc_length):
#             start = time.perf_counter()
#             best_pair = find_best_pair(spaced_chain)
#             spaced_chain = agglutinate_chain(best_pair, spaced_chain)
#             finish = time.perf_counter()
#             print(
#                 f"{i}: {''.join(best_pair)} - Computed in {round(finish - start,2)} secs"
#             )
#             if i in inter_save:
#                 vocab = count_units_in_chain(spaced_chain)
#                 print(
#                     f"Saving intermediate result. Vocabulary length: {len(vocab)-alpha_len}"
#                 )
#                 output_file = (
#                     f"voc_A_{corpus_name}_{len(vocab)-alpha_len}_inter"
#                 )
#                 util.dict2csv(vocab, output_file, paths.vocabularies)

#                 output_resume = f"{corpus_name}_resume"
#                 util.str2txt(spaced_chain, output_resume, paths.scratch)
#                 util.str2txt(
#                     f"Input corpus: {corpus_name}\nCorpus Lenght: {courpus_length}\nParallel: {parallelQ}\nNumbers of Cores: {n_cores}\nVocabulary length (so far): {len(vocab)-alpha_len}",
#                     f"{corpus_name}_resume_info",
#                     paths.scratch,
#                 )
#     print("Collecting frequencies of terms")
#     vocab = count_units_in_chain(spaced_chain)
#     if save_finalQ:
#         util.dict2csv(vocab, filename, paths.vocabularies)
#     return vocab

# def find_best_pair_par(chain_spaced):
#     pairs = Counter()
#     pre_units = chain_spaced.split()
#     for i in range(len(pre_units) - 1):
#         pairs[pre_units[i], pre_units[i + 1]] += 1
#     return dict(
#         pairs.most_common()[:100]
#     )  # Looking only on top 100 pre_units of each par list, for efficiency

# def build_vocab_par(
#     chain: str,
#     voc_length: int,
#     n_cores=4,
#     inter_save=[],
#     resumeQ = False,
#     save_finalQ=False,
#     filename = "voc_p_default",
#     corpus_name = "corp_deefault",
#     courpus_length = None,
#     ):
#     if resumeQ:
#         initial_chain = chain.replace(' ','').replace('\n','')
#     else:
#         initial_chain = chain
#     print("Spacing chain...")
#     spaced_chains = [" ".join(chain_part) for chain_part in util.partition(initial_chain, n_cores)]
#     print(f"Chain spaced")
#     alphas = util.multiprocessing(count_units_in_chain, spaced_chains)
#     alpha = dict(functools.reduce(operator.add, [Counter(voc) for voc in alphas]).most_common())
#     apha_len = len(alpha)
#     print(f'Length of initial alphabet: {apha_len}')
#     if resumeQ:
#         print("Resuming chain...")
#         spaced_chains = chain.split('\n')
#         print('Chain resumed')
#     if voc_length > 0:
#         print("Enter loop")
#         for i in range(voc_length):
#             start = time.perf_counter()
#             top_pairs_par = util.multiprocessing(find_best_pair_par, spaced_chains)
#             best_pair = functools.reduce(
#                 operator.add, [Counter(top_pairs) for top_pairs in top_pairs_par]
#             ).most_common()[0][0]
#             spaced_chains = util.multiprocessing(
#                 functools.partial(agglutinate_chain, best_pair), spaced_chains
#             )
#             finish = time.perf_counter()
#             print(
#                 f"{i}: {''.join(best_pair)} - Computed in {round(finish - start,2)} secs"
#             )
#             if i in inter_save:
#                 vocabs = util.multiprocessing(count_units_in_chain, spaced_chains)
#                 vocab = dict(
#                     functools.reduce(
#                         operator.add, [Counter(voc) for voc in vocabs]
#                     ).most_common()
#                 )
#                 print(f"Saving intermediate result. Vocabulary length: {len(vocab)-apha_len}")
#                 output_file = f"voc_A_{corpus_name}_p_{len(vocab)-apha_len}_inter"
#                 util.dict2csv(vocab, output_file, paths.vocabularies)

#                 output_resume = f"{corpus_name}_p_resume"
#                 util.str2txt('\n'.join(spaced_chains), output_resume, paths.scratch)
#                 util.str2txt(f'Input corpus: {corpus_name}\nCorpus Lenght: {courpus_length}\nParallel: {parallelQ}\nNumbers of Cores: {n_cores}\nVocabulary length (so far): {len(vocab)-apha_len}',f"{corpus_name}_p_resume_info",paths.scratch)
#     print('Collecting frequencies of terms')
#     vocabs = util.multiprocessing(count_units_in_chain, spaced_chains)
#     vocab = dict(
#         functools.reduce(operator.add, [Counter(voc) for voc in vocabs]).most_common()
#     )
#     if save_finalQ:
#         util.dict2csv(vocab, filename, paths.vocabularies)
#     return vocab

# def build_vocabulary(
#     chain,
#     printQ = True,
#     parallelQ = True,
#     n_cores = 4,
#     voc_len = 10,
#     inter_results = False,
#     resumeQ = False,
#     save_finalQ = False,
#     filename = "voc_default",
#     corpus_name = "corp_deefault",
#     courpus_length = None,
# ):
#     start = time.perf_counter()
#     print(f"Building Vocabulary")
#     if printQ:
#         print("Print is on")
#     else:
#         print("Print is off")
#     if parallelQ:
#         print(f"Method: Parallel - N° of Cores: {n_cores}")
#         voc = build_vocab_par(chain, voc_len, n_cores, inter_save=inter_results, save_finalQ=save_finalQ, resumeQ = resumeQ, filename = filename, corpus_name = corpus_name, courpus_length = courpus_length)
#     else:
#         print(f"Method: Sequential")
#         voc = build_vocab(chain, voc_len, inter_save=inter_results, save_finalQ=save_finalQ,resumeQ = resumeQ, filename = filename, corpus_name = corpus_name, courpus_length = courpus_length)
#     finish = time.perf_counter()
#     print(f"Vocabulary built in {round(finish - start,2)} secs")
#     return voc


# # 01.02 Segment Sentences #
# ###########################

# def load_vocabulary(
#     filename,
#     ):
#     # print(f"Loading Vocabulary ({os.path.basename(filename)})")
#     with open(f"{paths.vocabularies}{filename}.csv", "r") as f:
#         csv_reader = csv.reader(f)
#         voc_A = Counter()
#         for line in csv_reader:
#             voc_A[line[0]] = int(line[1])
#         voc_A = dict(voc_A)
#     # print('Done!\n')
#     return voc_A

# def load_test_sents(
#     filename
#     ):
#     print(f"Loading Test Sentences ({filename})")
#     with open(f"{paths.sentences}{filename}.txt", "r") as f:
#         sents = []
#         for line in f.readlines():
#             sents.append(line.rstrip())
#     print('Done!\n')
#     return sents

# # test_sent = "i have made my plans and i must stick to them"


# # def random_sent(sents):
# #     return sents[random.randrange(len(sents))]





# def tree_encoder(tree_data):
#     seg_tree = nx.DiGraph()
#     seg_tree.add_edges_from(tree_data)
#     string = list(seg_tree.nodes())[0][0]
#     gdf = {n: i for i, n in enumerate(seg_tree.nodes())}
#     gs = [(gdf[h], gdf[t]) for h, t in seg_tree.edges()]
#     node_interval = [n[1] for n in seg_tree.nodes()]
#     encoded_tree = (string, node_interval, gs)
#     return encoded_tree

# def forest_encoder(trees):
#     print("Encoding forest...")
#     start = time.perf_counter()
#     encoded_forest = util.multithreading(tree_encoder, trees)
#     finish = time.perf_counter()
#     print(f"Forest of {len(trees)} trees encoded in {round(finish-start,2)} secs")
#     return encoded_forest

# def save_forest(forest, filename):
#     print("Saving segmented trees")
#     encoded_forest = forest_encoder(forest)
#     util.list2csv(encoded_forest, filename, paths.segmentations)
#     print(f"Segmented chains saved as: {filename}")
#     return ''


# def segment_chainlist(
#     sents:list,
#     voc:dict,
#     seg_type = "sq",
#     weight_f = rankOnly,
#     zipf_factor = .135,
#     sample_size = 100,
#     randomQ = False,
#     parallelQ = True,
#     saveQ = False,
#     filename = "seg_default",
#     ):
#         assert seg_type in {"sq", "tr"}, "The segmentation type (seg_type) should be either 'sq' or 'tr'"

#         print("Computing segmented sequences")
#         start = time.perf_counter()
#         if sample_size==None:
#             sample_size = len(sents)
#         if randomQ:
#             chain_list = random.sample(sents,min(sample_size,len(sents)))
#         else:
#             chain_list = sents[:sample_size]
        
#         voc_rank = vocabulary_rank(voc,zipf_factor)

#         if seg_type == "sq":
#             print("Sequences always computed sequentially (no parallelism)")
#             segs = [" ".join(chain2seq(chain, voc, weight_f,voc_rank, zipf_factor)) for chain in chain_list]
            
#             print(f"Segmented chains (sequences) computed in:{time.perf_counter()-start} secs.")

#             if saveQ:
#                 print("Saving segmented sequences")
#                 util.str2txt("\n".join(segs),f"{seg_type}_{filename}",paths.segmentations)
#                 print(f"Segmented chains saved as: {filename}")
#             else:
#                 print("Segmented chains not saved.")
                
#         elif seg_type == "tr":
#             if parallelQ:
#                 print("Computing trees in parallel")
#                 segs = util.multiprocessing(functools.partial(chain2tree, voc=voc, weight_f=weight_f, v_rank = voc_rank, zipf_factor = zipf_factor), chain_list)
#             else:
#                 print("Computing trees sequentially")
#                 segs = [chain2tree(chain, voc, weight_f,voc_rank, zipf_factor) for chain in chain_list]

#             print(f"Segmented chains (trees) computed in:{time.perf_counter()-start} secs.")

#             if saveQ:
#                 save_forest(segs,f"{seg_type}_{filename}")
#             else:
#                 print("Segmented trees not saved.")


#         print("Done!\n")

#         return segs

# def plot_tree(tree_data, red=None, grey=None, fname=f"{paths.segment_graphs}seg_graph"):
#     seg_tree_graph = gv.Digraph(name=fname)
#     seg_tree = nx.DiGraph()
#     seg_tree.add_edges_from(tree_data)
#     tree_nodes_list = [(str(i), l) for l, i in list(seg_tree.nodes)]
#     for node in tree_nodes_list:
#         seg_tree_graph.node(
#             *node, color="white", fontsize="30", fontname="garamond"
#         )  # style="filled", color="grey")
#     seg_tree_graph.attr("edge", color="slategrey")
#     seg_tree_graph.edges([(str(p[1]), str(c[1])) for p, c in list(seg_tree.edges)])
#     if grey == None:
#         pass
#     else:
#         for node in [(str(i), l) for l, i in grey]:
#             if node in tree_nodes_list:
#                 seg_tree_graph.node(
#                     *node, style="dashed", color="grey", fontsize="30", fontname="garamond"
#                 )

#     if red == None:
#         pass
#     else:
#         for node in [(str(i), l) for l, i in red]:
#             if node in tree_nodes_list:
#                 seg_tree_graph.node(*node, color="red", fontsize="30", fontname="garamond")
#     return seg_tree_graph


# # 01.03 Extract Orthogonals #
# #############################


# def load_seqs(filename, n_start = None, n_end = None):
#     with open(f"{paths.segmentations}{filename}.txt", "r") as f:
#         seqs = []
#         for line in f.readlines():
#             seqs.append(line.rstrip())
#     return seqs[n_start:n_end]

# def tree_decoder(encoded_tree):
#     string = encoded_tree[0]
#     nodes_dict = {i: (string[n[0] : n[1]], n) for i, n in enumerate(encoded_tree[1])}
#     tree_data = [(nodes_dict[h], nodes_dict[t]) for h, t in encoded_tree[2]]

#     return tree_data

# def forest_decoder(encoded_trees):
#     print("Decoding trees...")
#     start = time.perf_counter()
#     decoded_trees = util.multiprocessing(tree_decoder, encoded_trees)
#     finish = time.perf_counter()
#     print(f"{len(decoded_trees)} trees decoded in {round(finish-start,2)} secs")
#     return decoded_trees

# def load_trees(filename, n_start = None, n_end = None, decodeQ = True):

#     loaded_trees = util.csv2list(filename, paths.segmentations, n_start, n_end)

#     if decodeQ:
#         output_forest = forest_decoder(loaded_trees)
#     else:
#         output_forest = loaded_trees
    
#     return output_forest

# def load_segs(
#     filename,
#     n_start = None,
#     n_end = None
#     ):
#     seg_type = filename[:2]
#     assert seg_type in {"sq","tr"}, "The type of the segmentation file does not seem to match the possible data types ('sq' and 'tr')"

#     start = time.perf_counter()
#     print(f"Loading segmentations from: {filename}")
#     if seg_type=="sq":
#         seg_type_name = "sequences"
#         segs = load_seqs(filename, n_start, n_end)
#     elif seg_type=="tr":
#         seg_type_name = "trees"
#         segs = load_trees(filename, n_start, n_end)

#     finish = time.perf_counter()
#     print(f"{len(segs)} {seg_type_name} loaded in {round(finish-start,2)} secs")
#     print('Done!\n')
#     return segs

# def extract_bigrams_from_chains(list_of_chains):
#     bigrams = []
#     for chain in list_of_chains:
#         bigrams += util.subsequences(chain.split(), 2)
#     bg_dict = dict(Counter(bigrams).most_common())
#     return bg_dict


# def extract_siblings(tree_data, Llen = [], Rlen = []):

#     tree = nx.DiGraph()
#     tree.add_edges_from(tree_data)

#     neighbors = [tuple([n[0] for n in tree.neighbors(node)]) for node in tree.nodes()]
#     siblings = []
#     for p in neighbors:
#         if  p != ():
#             siblings.append(p)
#     return siblings

# def bulk_extract_siblings(forest):
    
#     siblings = util.multithreading(extract_siblings,forest)
#     siblings_flat = list(itertools.chain(*siblings))

#     siblings_dict = Counter(siblings_flat)

#     return dict(siblings_dict.most_common())


# def save_orthogonals(orthogonals, filename, freq_min = 0):
#     if not os.path.isdir(paths.orthogonals):
#         os.makedirs(paths.orthogonals)
#     with open(f"{paths.orthogonals}{filename}_{freq_min}.csv", "w") as nf:
#         for key, value in orthogonals.items():
#             if value >= freq_min:
#                 nf.write(f"{key[0]},{key[1]},{value}\n")


# def extract_orthogonals(
#     list_of_segs,
#     saveQ = False,
#     filename = "ortho_default",
#     freq_min = 0,
#     ):
#     print('Extracting orthogonals...')
#     seg_type = type(list_of_segs[0])
#     assert seg_type in {str,list}, "Format of segmentations not recognized"

#     start = time.perf_counter()
#     if seg_type == str:
#         print('Extracting bigrams from sequences...')
#         orthos = extract_bigrams_from_chains(list_of_segs)
#     elif seg_type == list:
#         print('Extracting siblings from trees...')
#         orthos = bulk_extract_siblings(list_of_segs)
#     finish = time.perf_counter()
#     print(f"Orthogonals of {len(list_of_segs)} {'sequences' if seg_type==str else 'trees'} extracted in {round(finish-start,2)} secs")
#     if saveQ:
#         if type(freq_min) == list:
#             for n in freq_min:
#                 save_orthogonals(orthos,filename,n)
#                 print(f"Orthogonals saved as {filename}_{n}")
#         else:
#             save_orthogonals(orthos,filename,freq_min)
#             print(f"Orthogonals saved as {filename}_{freq_min}")
#     else:
#         print("Orthogonals not saved")
#     print('Done\n')
#     return orthos


# # #########################
# # # 02 - TYPING FUNCTIONS #
# # #########################


# def load_orthogonality_data(
#     filename,
#     freq_min = 0,
#     ):
#     print(f"Loading data... ({os.path.basename(filename)})")
#     print(f"Loading orthogonals of frequency equal or higher than: {freq_min}")
#     with open(f"{paths.orthogonals}{filename}.csv", "r") as f:
#         csv_reader = csv.reader(f)
#         biGrams_all = Counter()
#         for line in csv_reader:
#             if int(line[-1]) >= freq_min:
#                 biGrams_all[tuple([line[i] for i in range(len(line)-1)])] = int(line[-1])
#     print(f"Orthogonals final length: {len(biGrams_all)}")
#     print(f"Done!\n")
#     return dict(biGrams_all.most_common())

# def build_terms_contexts(
#     orthogonals,
#     hand_picked_terms=[],
#     hand_picked_contexts=[],
#     trim_terms=None,
#     trim_contexts=None,
#     symmetricQ=False):
#     # TODO: There should be a much faster way of constructing this matrix by extracting the info of the dic and directly building a csr_matrix out of it
#     print("Building Terms and Contexts...")
#     terms_dict = util.marginalize(orthogonals,"left")
#     contexts_dict = util.marginalize(orthogonals,"right")
#     if symmetricQ in ('i','u'):
#         terms_context_union = Counter(terms_dict)+Counter(contexts_dict)
#         terms = contexts = [k for k,v in terms_context_union.most_common()]
#         if symmetricQ == "i":
#             terms_contexts_i = set(list(terms_dict.keys())).intersection(set(list(contexts_dict.keys())))
#             terms = contexts = [t for t in terms if t in terms_contexts_i]
#         if trim_terms==None:
#             print('No Trim')
#         else:
#             print(
#                 f"Trim terms and contexts: {trim_terms}"
#             )
#             terms = contexts = terms[:trim_terms]

#     else:
#         terms = [k for k,v in Counter(terms_dict).most_common()]
#         contexts = [k for k,v in Counter(contexts_dict).most_common()]
#         if trim_terms==None:
#             print('No Term Trim')
#         else:
#             print(
#                 f"Trim terms: {trim_terms}"
#             )
#             terms = terms[:trim_terms]
#         if trim_contexts==None:
#             print('No Context Trim')
#         else:
#             print(
#                 f"Trim Contexts: {trim_contexts}"
#             )
#             contexts = contexts[:trim_contexts]

#     if len(hand_picked_terms) > 0:
#         terms = hand_picked_terms
#     if len(hand_picked_contexts) > 0:
#         contexts = hand_picked_contexts
#     print(f"Terms length: {len(terms)}")
#     print(f"Contexts length: {len(contexts)}")
#     print("Done\n")
#     return [terms, contexts]


# def mm_no_modif(contexts, orthogonals, term):
#     return [
#         orthogonals.get((term, context),0) for context in contexts
#     ]

# def matrix_maker(
#     terms,
#     contexts,
#     orthogonals,
#     measure = mm_no_modif):
#     results = util.multithreading(functools.partial(measure, contexts, orthogonals), terms)
#     return results

# def build_term_context_matrix(
#     terms,
#     contexts,
#     orthogonals,
#     normalizeQ = False):
#     print("Building oR Matrix...")
#     start = time.perf_counter()
#     if normalizeQ:
#         orthogonals = util.normalize_dict(orthogonals)
#     matrix = csr_matrix(matrix_maker(terms,contexts,orthogonals))
#     finish = time.perf_counter()
#     print(f"Term-Context Matrix built in {round(finish-start,2)} secs.\n")
#     return matrix

# def build_pmi_matrix(
#     term_context_matrix,
#     type = "pmi",
#     alpha = .75,
#     normalizeQ = False,
#     ):

#     print("Computing PMI Matrix...")
#     print(f"Type: {type}")
#     if "s" in type:
#         print(f"Smoothing (alpha): {alpha}")
#     start = time.perf_counter()
#     pmi_matrix = util.pmi(term_context_matrix,alpha=alpha,type_pmi=type)
#     finish = time.perf_counter()
#     if normalizeQ:
#         print("Normalizing Matrix")
#         pmi_matrix = (1/(pmi_matrix.sum()))*pmi_matrix
#     print(f"PMI Matrix built in {round(finish-start,2)} secs.")
#     print("Done\n")
#     return pmi_matrix

# def cut_thres(row, thres):
#     ortho_index = (row > thres) * 1
#     return ortho_index

# def cut_mean(row, thres):
#     if len(row.data) == 0:
#         ortho_index = row
#     else:
#         mean_cut = row.data.mean()
#         if mean_cut == row.data.max():
#             mean_cut = mean_cut / 2
#         mean_cut = row.data.mean()*thres
#         ortho_index = (row > mean_cut) * 1
#     return ortho_index

# def build_bin_matrix(
#     matrix,
#     cut_func = "cut_thres",
#     thres = 1e-5,
#     parallelQ = False
#     ):
#     """
#     In general, parallelizing in this case is slower than the sequential method (and multiprocessing slower than multithreading). This should still tried out. Maybe other ways of parallelizing are possible.
#     """
#     print(f"Binarizing Matrix...")
#     print(f"Cut Function: {cut_func}")
#     cut_func_eval = eval(cut_func)
#     start = time.perf_counter()
#     if parallelQ:
#         print("Method: Parallel")
#         disc_rows = util.multithreading(
#             functools.partial(cut_func_eval, thres=thres), matrix
#         )
#     else:
#         print("Method: Sequential")
#         disc_rows = []
#         for row in matrix:
#             disc_row = cut_func_eval(row, thres)
#             disc_rows.append(disc_row)
#     finish = time.perf_counter()
#     print(f"Binary matrix built in {round(finish-start,2)} secs.\n")
#     return vstack(disc_rows).astype(np.bool)

# def reduce_matrix(
#     matrix,
#     reduce_thres = 2):
#     """
#     Remove rows from binary matrix that have less units than a given threeshold (min_len)
#     """
#     print(f"Reducing matrix. Threshold: {reduce_thres}")
#     ones_factor = np.ones(matrix.shape[1],dtype=np.int8)
#     len_test = matrix.dot(ones_factor)
#     indices = np.where(len_test>=reduce_thres)
#     result = matrix[indices]
#     print(f"New reduced shape: {result.shape}")
#     return result

# def indiv_intersection(set_of_sets, reduce_thres, indiv_set):
#     inter_set = {indiv_set.intersection(s) for s in set_of_sets}
#     inter_set = {s for s in inter_set if len(s)>=reduce_thres}
#     inter_set = inter_set-set_of_sets
#     return inter_set

# def powerset_M(
#     matrix,
#     reduce_thres = 2,
#     bound = 20,
#     parallel=False,
#     ):
#     """
#     Takes a Boolean matrix and computes a matrix whose rows are the conjunction of all possible combinations of the rows of the original matrix (the conjunction of all the elements of the powerset of the set of original rows).
    
#     Rows with strictly less components than the integer defined as "reduce_thres" are disregarded.
    
#     The option "bound" establishes a limit of cardinality for the elements of the powerset to be computed.

#     Parallelism seems to slow down the computation

#     The final order of the rows in the matrix is not handled
#     """
#     start = time.perf_counter()



#     def row_sets_to_matrix(row_as_set:tuple):
#         r_index = row_as_set[0]
#         r_set = row_as_set[1]
#         matrix_row = [[r_index,c] for c in r_set]
#         return matrix_row

#     def sets2matrix(set_of_sets, width, parallel=False):
#         """
#         Converts a set of frozen sets into a sparse matrix in which each frozenset is a row and its elements are indices of columns. The order of the rows is not handled.
#         """
#         if parallel:
#             print("Building matrix in parallel")
#             csr_indices = util.multithreading(row_sets_to_matrix,enumerate(list(set_of_sets)))
#         else:
#             csr_indices = [row_sets_to_matrix(row) for row in enumerate(list(set_of_sets))]
#         csr_indices = np.array(util.flatten(csr_indices))
#         data = np.ones(len(csr_indices))
#         ps_M = csr_matrix((data, (csr_indices[:,0], csr_indices[:,1])), shape=(len(set_of_sets),width), dtype=bool)
#         return ps_M

#     print(f"Input size:\t{matrix.shape}")
#     matrix = reduce_matrix(matrix, reduce_thres)

#     print("Computing the conjunction of all the subsets of rows from matrix")
#     print(f"Parallel:\t{parallel}")

#     n = 1
#     # M_length = matrix.shape[0]
    
#     print("Transforming matrix into set of sets...")
#     matrix_dict = defaultdict(list)
#     for i, j in zip(*matrix.nonzero()):
#         matrix_dict[i].append(j)

#     matrix_as_sets = {frozenset(i) for i in matrix_dict.values()}


#     n_subset_M = matrix_as_sets
#     collect_n_subsets = matrix_as_sets

#     while n<bound+1:

#         n += 1
#         print(f"Subsets of size: {n}")

#         start_st = time.perf_counter()
        
#         if parallel:
#             n_subset_M = util.multithreading(functools.partial(indiv_intersection, n_subset_M, reduce_thres), matrix_as_sets)
#         else:
#             n_subset_M = [indiv_intersection(n_subset_M, reduce_thres,row) for row in matrix_as_sets]     

#         print(f"****\t Step computed in {round(time.perf_counter()-start_st,2)} secs. \t****")

#         start_st = time.perf_counter()
#         # n_subset_M = util.sp_unique(csr_matrix(vstack(n_subset_M_n)))
#         n_subset_M = set.union(*n_subset_M)
#         print(f"****\t Step computed in {round(time.perf_counter()-start_st,2)} secs. \t****")


#         start_st = time.perf_counter()
#         n_subset_M = n_subset_M-collect_n_subsets
#         print(f"****\t Step computed in {round(time.perf_counter()-start_st,2)} secs. \t****")
#         print(f"New:\t{len(n_subset_M)}")


#         if len(n_subset_M)==0:
#             print(f"\nEmpty subsets from subsets greater than {n-1} elements.\n")
#             break
#         else:
#             start_st = time.perf_counter()
#             collect_n_subsets = collect_n_subsets.union(n_subset_M)
#             print(f"Stacked:\t{len(collect_n_subsets)}")
#             print(f"****\t Step computed in {round(time.perf_counter()-start_st,2)} secs. \t****")

#         print("\n")

#     print("Transforming set of sets into matrix...")
#     start_st = time.perf_counter()
#     ps_M = sets2matrix(collect_n_subsets, matrix.shape[1],parallel=parallel)
#     print(f"Matrix built in {round(time.perf_counter()-start_st,2)} secs.")


#     finish = time.perf_counter()
#     print(f"Powerset matrix computed in {round(finish-start,2)} secs.")
#     print(f"Final size:\t{ps_M.shape}")
#     print("=================================\n")

#     return ps_M

# def powerset_M_parallel(
#     matrix_list:list,
#     reduce_thres = 2,
#     bound = 20,
#     ):
#     """
#     Computes the powerset of several matrices in parallel. Matrices need to be given in a list.
#     """
#     print("Computing Power Set Matrices...\n")
#     st = time.perf_counter()
#     matrix_list_ps = util.parallel_processes(
#         [
#             [functools.partial(powerset_M,
#             reduce_thres=reduce_thres,
#             bound=bound,
#             parallel=False),matrix]
#             for matrix in
#             matrix_list]
#     )
#     fin = time.perf_counter()
#     print(f"Powerset of {len(matrix_list)} matrices computed in parallel in {round(fin-st,2)} secs.\n")
#     return matrix_list_ps


# def orthogonal_row(row, ooQ):
#     if row.sum() == 0:
#         biorthos_vec = csr_matrix((1, ooQ.shape[1]), dtype=np.int8)
#     else:
#         indices = row.indices
#         ortho_rows = ooQ[indices, :]
#         biorthos_vec = ortho_rows.min(axis=0)
#     return biorthos_vec

# def biorthogonal(
#     oQ,
#     ooQ,
#     parallelQ=False
#     ):
#     # Parallelization is slower here
#     """
#     Given two corresponding binary matrixes of orthogonality (ex: to the Right [oQ] and back to the Left of that Right [ooQ]), this function computes the biorthogonal terms for each row of oQ (NB: each row might correspond to the orthogonals of multiple terms if initial matrix results form powerset_M). The resulting matrix is no longer a term-context matrix, but a terms-term (the class of terms out of which the bi-orthogonal types are composed belong to the same set of terms as the class of terms generating it)
#     """
#     print("Computing Biorthogonal Matrix...")
#     start = time.perf_counter()
#     if parallelQ:
#         print("Method: Parallel")
#         row_list = [r for r in oQ]
#         biorthos = util.multithreading(
#             functools.partial(orthogonal_row, ooQ=ooQ), row_list
#         )
#     else:
#         print("Method: Sequential")
#         biorthos = []
#         for row in oQ:
#             biorthos_vec = orthogonal_row(row, ooQ)
#             biorthos.append(biorthos_vec)
#     finish = time.perf_counter()
#     print(f"Biorthogonal Matrix computed in {round(finish-start,2)} secs.\n")
#     return csr_matrix(vstack(biorthos),dtype=np.bool)

# def biorthogonal_parallel(
#     list_pairs_matrices:list
# ):
#     """
#     Computes the biorthogonal of several (pairs of) matrices in parallel. Pairs of matrices need to be given in a list.
#     """
#     print("Computing biorthogonal matrices...")
#     bo_M = util.parallel_processes(
#         [
#             [biorthogonal,o,oo]
#             for o,oo in
#             list_pairs_matrices
#             ]
#     )
#     return bo_M

# def bo_Types(
#     terms,
#     contexts,
#     bo_M,
#     o_M,
#     min_len=2,
#     saveQ = False,
#     filename = "bo_o_",
#     ):
#     """
#     bo_M and o_M should be of shape (n,len(terms)) and (n,len(contexts))
#     if filter_sT = True, it filters out those couples whose ortho type is included in that of another couple of the list for the same bo type
#     """
#     print('Building bo Types List...')
#     assert bo_M.shape[0]==o_M.shape[0], "ERROR: bo_M and o_M are not of same length"
#     start = time.perf_counter()
#     types_list = []
#     for index in range(bo_M.shape[0]):
#         types = (
#             tuple(sorted([terms[i] for i in bo_M[index].indices])),
#             tuple(sorted([contexts[i] for i in o_M[index].indices]))
#         )
#         types_list.append(types)

#     # First possible structural decission: Exclude orthogonal relations involving types of length smaller than a given n (min_len) (typically 2)
#     # Possible objection: There could be categories with unmarked members (for instance the singular in English, with "s" as mark of the plural, or "th" as mark of ordinal)
#     print(f"Filtering to types of length equal or greater than {min_len}")
#     types_list = [(l,r) for l,r in types_list if min(len(l), len(r)) >= min_len]

#     finish = time.perf_counter()
#     print(f"Types List built in {round(finish-start,2)} secs.\n")

#     if saveQ:
#         util.list2csv(
#             types_list,
#             filename,
#             paths.types
#             )
#     print(f"Types table saved as: {filename}")

#     return types_list


# def build_types_parallel(
#     quad_t_c_bo_o_list:list,
#     min_len = 2,
#     saveQ = False,
#     filename = "bo_o_",
#     ):
#     """
#     Computes the biorthogonal of several matrices in parallel. lists of [terms, contexts, bo-matrix and o-matrix] need to be given in a list.
#     """
#     bo_oR_ = util.parallel_processes(
#         [
#             [functools.partial(bo_Types,min_len=min_len, saveQ=saveQ, filename=filename+"_"+str(n+1)),t,c,bo,o]
#             for n,(t,c,bo,o) in
#             enumerate(quad_t_c_bo_o_list)
#         ]
#     )
#     return bo_oR_

# def load_type_list(
#     filename,
#     n_start = None,
#     n_end = None,
#     ):
#     def csv2T_list(filename: str, directory, n_start=None, n_end=None):
#         with open(directory+filename, "r") as f:
#             csv_reader = csv.reader(f)
#             my_list = [
#                 (ast.literal_eval(lT), ast.literal_eval(rT))
#                 for lT,rT in util.take(n_end, csv_reader)
#             ]
#         return my_list[n_start:n_end]
#     files = [fn for fn in os.listdir(paths.types) if filename in fn]
#     type_lists = [
#         csv2T_list(fn,paths.types,n_start,n_end)
#         for fn in files
#     ]
#     return type_lists



# def subtype_rels(type_list, raw = True):
#     """
#     Builds a list of tuples (i,j) of types, where j is a subtype of i
#     Set raw = True for raw list of types (list of tuples). If False, the function takes the first element of each item of the list (used for building graphs)
#     """
#     # type_list = set(type_list)
#     lattice_edges = []
#     for i in type_list:
#         for j in type_list:
#             if raw:
#                 if set(i).issubset(set(j)) and i != j:
#                     lattice_edges.append((j, i))
#             else:
#                 if set(i[0]).issubset(set(j[0])) and i[0] != j[0]:
#                     lattice_edges.append((j, i))
#     return lattice_edges

# def filter_biortho_subtypes(types_list):
#     """
#     Given a list of couples of bo-o types, it filters out those couples whose ortho type is included in that of another couple of the list for the same bo type
#     """
#     types_list_filter_dict = defaultdict(list)
#     for boT,oT in types_list:
#         types_list_filter_dict[boT].append(oT)
#     reduced_types_list = []
#     for boT in (i for i in types_list_filter_dict.keys() if len(i)>1):
#         oT_indiv = util.delete_duplicates(types_list_filter_dict[boT])
#         oT_st = subtype_rels(oT_indiv)
#         oT_bot = [b for t,b in oT_st]
#         reduced_types_list_indiv = [(boT,oT) for oT in oT_indiv if oT not in oT_bot]
#         reduced_types_list.extend(reduced_types_list_indiv)
#     return reduced_types_list

# def build_ortho_graph(
#     bo_types_list,
#     side = "R",
#     ortho=True,
#     subtype_filter=True,
#     filter_sT = False,
#     verbose=True
#     ):
    
#     print('Building Graph...') if verbose else None

#     if filter_sT:
#         print("Filtering orthogonal subtypes")
#         bo_types_list = filter_biortho_subtypes(bo_types_list)

#     print('Building orthogonal relations') if verbose else None

#     if side == "R":
#         boT_side = "L"
#         oT_side = "R"
#     else:
#         boT_side = "R"
#         oT_side = "L"

#     ortho_rels = [
#         ((l, "bo", boT_side), (r, "o", oT_side))
#         for l, r in bo_types_list        
#     ]

#     # Second structural decission: Exclude orthogonal relations whose terms don't involve subtyping relations with other terms in their respective sides
#     # Possible objection: The search for types through orthogonality is not directly related to subtyping (ex: what should be the subtyping structure to which "ed,ing,s" belongs?)

#     if subtype_filter:
#         print('Building L, R, o and bo subtyping filters') if verbose else None

#         struct_elements_boT = []
#         for i,j in subtype_rels([l for l, r in ortho_rels], raw = False):
#             struct_elements_boT += [i,j]
#         struct_elements_boT = set(struct_elements_boT)

#         struct_elements_oT = []
#         for i,j in subtype_rels([r for l, r in ortho_rels], raw = False):
#             struct_elements_oT += [i,j]
#         struct_elements_oT = set(struct_elements_oT)

#         print('Filtering orthogonalities to subtypes structures') if verbose else None

#         struct_ortho_rels = [
#             (i, j)
#             for i, j in ortho_rels
#             if i in struct_elements_boT and j in struct_elements_oT
#         ]
#     else:
#         struct_ortho_rels = ortho_rels

#     print('Constructing transitive reduction of lattices') if verbose else None

#     tred_graph_boT = nx.DiGraph()
#     tred_graph_boT.add_edges_from(subtype_rels([l for l, r in struct_ortho_rels], raw = False))
#     tred_graph_boT = nx.algorithms.dag.transitive_reduction(tred_graph_boT)

#     tred_graph_oT = nx.DiGraph()
#     tred_graph_oT.add_edges_from(subtype_rels([r for l, r in struct_ortho_rels], raw = False))
#     tred_graph_oT = nx.algorithms.dag.transitive_reduction(tred_graph_oT)

#     print('Merging boT and oT graphs') if verbose else None
#     complete_graph = functools.reduce(
#         nx.compose, [tred_graph_boT, tred_graph_oT]
#     )

#     print('Adding orthogonal edges') if verbose else None
#     if side == "L":
#         struct_ortho_rels = [(r,l) for l,r in struct_ortho_rels]
#     if ortho:
#         complete_graph.add_edges_from(struct_ortho_rels, edge_type="ortho")
#     print("Done!\n") if verbose else None
#     return complete_graph

# def compose_graphs(list_of_graphs: list):
#     return functools.reduce(
#         nx.compose, list_of_graphs
#     )

# def make_label(label: tuple, LRN="N"):
#     """
#     Given a list of tuples corresponding to Types, it outputs strings for names of those
#     Types in the graph, inserting linebrakes if the string is too long.
#     Options for making L or R types with 'LRN = ': 'L', 'R', 'LR'.
#     """
#     if len(label) < 6:
#         lab = str(label)
#     elif len(label) > 15:
#         lab_cut = str(label[:15])
#         groups = str(lab_cut).split(" ")
#         lab = "\n".join(
#             [
#                 " ".join(groups[:5]),
#                 " ".join(groups[6:10]),
#                 " ".join(groups[11:15]) + "... " + "[+" + str(len(label) - 15) + "]",
#             ]
#         )
#     else:
#         n = int(len(label) / 2)
#         groups = str(label).split(" ")
#         lab = "\n".join([" ".join(groups[:n]), " ".join(groups[n:])])
#     if LRN == "L":
#         return lab.replace("'", "").replace("(", "").replace(")", "") + " /"
#     elif LRN == "R":
#         return "\\\\ " + lab.replace("'", "").replace("(", "").replace(")", "")
#     elif LRN == "LR":
#         return "\\\\ " + lab.replace("'", "").replace("(", "").replace(")", "") + " /"
#     else:
#         return lab.replace("'", "").replace("(", "").replace(")", "")

# def simple_lattice_graph(
#     lattice_nx_graph,
#     fname="LatGraph",
#     fdir=paths.lattice_graphs,
#     colors_list=[
#         "burlywood2",
#         "blanchedalmond",
#         "grey80",
#         "slategrey",
#     ],
#     font="RobotoMono-Thin"
# ):
#     print("Plotting graph...")

#     lattice_g = gv.Digraph(name=fname, directory=fdir, strict=True)

#     lattice_g.attr("node", fontname = font)
#     lattice_g.attr("edge", color="grey80")

#     lattice_g.edges(
#         [(str(i), str(j)) for i, j in lattice_nx_graph.edges()]
#     )

#     for node in lattice_nx_graph.nodes:
#         lattice_g.node(
#             str(node),
#             make_label(node),
#             style="filled",
#             colorscheme = "OrRd9",
#             color=len(node),
#             shape="oval",
#             #orientation="0" if node[-1] == "L" else "90",
#             #margin="0,0" if node[0] in LR_intersect else "0.2,0.2",
#         )
#     print("Finished!")
#     return lattice_g

# def plot_lattice_graph(raw_parads,filter_single_nodes = False):
#     lattice_rels = subtype_rels(raw_parads)
#     tred_graph = nx.DiGraph()
#     tred_graph.add_edges_from(lattice_rels)
#     if not filter_single_nodes:
#         tred_graph.add_nodes_from(raw_parads)
#     tred_graph = nx.algorithms.dag.transitive_reduction(tred_graph)

#     return simple_lattice_graph(tred_graph)

# def lattice_graph(
#     complete_nx_graph,
#     fname="Graph",
#     fdir=paths.lattice_graphs,
#     colors_list=[
#         "burlywood2",
#         "blanchedalmond",
#         "grey80",
#         "slategrey",
#     ],
#     dependenciesQ=True,
#     LRQ = True,
#     font="RobotoMono-Thin"
# ):
#     print("Plotting graph...")
#     LR_intersect = set([n[0] for n in complete_nx_graph.nodes() if n[-1] == 'L']) & set([n[0] for n in complete_nx_graph.nodes() if n[-1] == 'R'])

#     color_switch = {
#         ("o","L") : 1,
#         ("bo","R") : 0,
#         ("bo","L") : 0,
#         ("o","R") : 1,
#         ("obo","L") : 2,
#         ("obo","R") : 2,
#     }

#     Complete_LR = gv.Digraph(name=fname, directory=fdir, strict=True)

#     # top_bottom = {("⊤",), ("⊥",)}
#     Complete_LR.attr("node", fontname = font)
#     Complete_LR.attr("edge", color="grey80")

#     Complete_LR.edges(
#         [(str(i), str(j)) for i, j, d in complete_nx_graph.edges(data=True) if d == {}]
#     )

#     for node in complete_nx_graph.nodes:
#         # if node[1] == "o":
#         Complete_LR.node(
#             str(node),
#             make_label(node[0], LRN=node[2]),
#             style="filled",
#             color=colors_list[color_switch[node[1:3]]],
#             shape="doubleoctagon" if node[0] in LR_intersect else "cds",
#             orientation="0" if node[-1] == "L" else "90",
#             margin="0,0" if node[0] in LR_intersect else "0.2,0.2",
#         )

#     if dependenciesQ:
#         Complete_LR.attr("edge", arrowhead="none", color="darksalmon")
#         Complete_LR.edges(
#             [
#                 (str(i), str(j))
#                 for i, j, d in complete_nx_graph.edges(data=True)
#                 if d != {}
#             ]
#         )
#     print("Finished!")
#     return Complete_LR

# def bigram_types(bigram, types_lists, separate_lists=False):
#     possible_types = []
#     for list in types_lists:
#         ortho_types = [(lT, rT) for lT, rT in list if bigram[0] in lT and bigram[1] in rT]
#         if separate_lists:
#             possible_types.append(ortho_types)
#         else:
#             possible_types.extend(ortho_types)
#     return possible_types


# # Draft functions
# from transformers import pipeline

# unmasker = pipeline('fill-mask', model='distilbert-base-uncased',top_k=10)

# def sent_paradigm(sentence,thres=0):
#     sent_list = sentence.split()
#     sent_mask = [" ".join([token if n!=i else "[MASK]" for n,token in enumerate(sent_list)]) for i in range(len(sent_list))]
#     parads = [{i['token_str']:i['score'] for i in unmasker(sent) if i['score']>thres and i['token_str'] not in {
#         ".",":",",","…","'","’","′",'"',"•",";","`","-","“","...","?","!","/","&","–"
#         } } for sent in sent_mask]

#     return (parads)

# def extract_parads(sents_sample,thres=.01): 

#     parads = Counter()
#     i=0
#     print_i = [i*10 for i in range(len(sents_sample))]
#     for sent in sents_sample:
#         i+=1
#         if i in print_i:
#             print(f"{i}/{len(sents_sample)}")
#         sent_parad, probability_mass = sent_paradigm(sent,thres=thres,prob=False)
#         for p in sent_parad:
#             if len(p)>1:
#                 parads[tuple(sorted(p))]+=1
#     return parads


# def simple_lattice_graph(
#     lattice_nx_graph,
#     fname="LatGraph",
#     fdir=paths.lattice_graphs,
#     colors_list=[
#         "burlywood2",
#         "blanchedalmond",
#         "grey80",
#         "slategrey",
#     ],
#     font="RobotoMono-Thin"
# ):

#     lattice_g = gv.Digraph(name=fname, directory=fdir, strict=True)

#     lattice_g.attr("node", fontname = font)
#     lattice_g.attr("edge", color="grey80")

#     lattice_g.edges(
#         [(str(i), str(j)) for i, j in lattice_nx_graph.edges()]
#     )
#     max_node = max([node[1] for node in lattice_nx_graph.nodes])
#     for node in lattice_nx_graph.nodes:
#         node_1_9_ranking = max(1,int(node[1]*9/max_node))
#         lattice_g.node(
#             str(node),
#             make_label(node[0]),
#             style="filled",
#             colorscheme = "oranges9",
#             color = str(node_1_9_ranking),
#             fontcolor = "white" if node_1_9_ranking>7 else "black",
#             shape= "doubleoctagon" if node[1]==max_node else "oval",
#             #orientation="0" if node[-1] == "L" else "90",
#             #margin="0,0" if node[0] in LR_intersect else "0.2,0.2",
#         )
#     return lattice_g

# def plot_lattice_graph(raw_parads,filter_single_nodes = False):
#     lattice_rels = subtype_rels(raw_parads,raw=False)
#     tred_graph = nx.DiGraph()
#     tred_graph.add_edges_from(lattice_rels)
#     if not filter_single_nodes:
#         tred_graph.add_nodes_from(raw_parads)
#     tred_graph = nx.algorithms.dag.transitive_reduction(tred_graph)

#     return simple_lattice_graph(tred_graph)
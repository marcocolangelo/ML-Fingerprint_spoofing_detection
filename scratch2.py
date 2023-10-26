import re
results=[]
results6 = []
resultsNone = []
gmm_pca6_globfull_DiagAndTied = {('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=1 pca = 6: 0.32237704918032783 znorm: False', 0.32237704918032783): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=1 pca = 6: 0.2580327868852459 znorm: False', 0.2580327868852459): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=1 pca = 6: 0.25834016393442627 znorm: False', 0.25834016393442627): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.3030532786885246 znorm: False', 0.3030532786885246): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.2493032786885246 znorm: False', 0.2493032786885246): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.2589754098360656 znorm: False', 0.2589754098360656): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=1 pca = 6: 0.3355122950819672 znorm: False', 0.3355122950819672): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=1 pca = 6: 0.2755327868852459 znorm: False', 0.2755327868852459): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=1 pca = 6: 0.2589754098360656 znorm: False', 0.2589754098360656): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.34049180327868855 znorm: False', 0.34049180327868855): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.26774590163934425 znorm: False', 0.26774590163934425): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.25364754098360653 znorm: False', 0.25364754098360653): None}.keys()
gmm_pcaNone_globfull_DiagAndTied = {('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=1 pca = None: 0.3739549180327869 znorm: False', 0.3739549180327869): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=1 pca = None: 0.2730122950819672 znorm: False', 0.2730122950819672): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=1 pca = None: 0.25241803278688524 znorm: False', 0.25241803278688524): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.3789754098360656 znorm: False', 0.3789754098360656): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.28051229508196723 znorm: False', 0.28051229508196723): None, ('GMM min_DCF mode_target=full e mode_not_target=diag con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2596106557377049 znorm: False', 0.2596106557377049): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=1 pca = None: 0.3314549180327869 znorm: False', 0.3314549180327869): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=1 pca = None: 0.2646106557377049 znorm: False', 0.2646106557377049): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=1 pca = None: 0.2561475409836066 znorm: False', 0.2561475409836066): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.33331967213114755 znorm: False', 0.33331967213114755): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.2518032786885246 znorm: False', 0.2518032786885246): None, ('GMM min_DCF mode_target=full e mode_not_target=tied con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.25209016393442624 znorm: False', 0.25209016393442624): None}.keys()
gmm_pca6_globfull = {('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=1 pca = 6: 0.27676229508196726 znorm: False', 0.27676229508196726): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=1 pca = 6: 0.25647540983606554 znorm: False', 0.25647540983606554): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=1 pca = 6: 0.2655327868852459 znorm: False', 0.2655327868852459): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=2 pca = 6: 0.2752254098360656 znorm: False', 0.2752254098360656): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=2 pca = 6: 0.25553278688524594 znorm: False', 0.25553278688524594): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=2 pca = 6: 0.25772540983606557 znorm: False', 0.25772540983606557): None}
gmm_pcaNone_globfull = {('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=1 pca = None: 0.2830327868852459 znorm: False', 0.2830327868852459): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=1 pca = None: 0.25209016393442624 znorm: False', 0.25209016393442624): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=1 pca = None: 0.2633401639344263 znorm: False', 0.2633401639344263): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=2 t_max_n=2 pca = None: 0.27961065573770494 znorm: False', 0.27961065573770494): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=4 t_max_n=2 pca = None: 0.2602049180327869 znorm: False', 0.2602049180327869): None, ('GMM min_DCF mode_target=full e mode_not_target=full con K = 5 , nt_max_n=8 t_max_n=2 pca = None: 0.2646311475409836 znorm: False', 0.2646311475409836): None}

gmm_pca6_globfull.update(gmm_pca6_globfull_DiagAndTied)

for string in gmm_pca6_globfull:
    string = string[0]
    result = {}
    result['mode_target'] = re.search(r'mode_target=(\w+)', string).group(1)
    result['mode_not_target'] = re.search(r'mode_not_target=(\w+)', string).group(1)
    result['nt_max_n'] = int(re.search(r'nt_max_n=(\d+)', string).group(1))
    result['t_max_n'] = int(re.search(r't_max_n=(\d+)', string).group(1))
    pca_match = re.search(r'pca\s*=\s*(\d+)', string)
    if pca_match:
        result['pca'] = int(pca_match.group(1))
    else:
        result['pca'] = None
    result['DCF_min'] = float(re.search(r':\s*([\d.]+)', string).group(1))
    results.append(result)
    
for string in gmm_pcaNone_globfull_DiagAndTied:
    string = string[0]
    result = {}
    result['mode_target'] = re.search(r'mode_target=(\w+)', string).group(1)
    result['mode_not_target'] = re.search(r'mode_not_target=(\w+)', string).group(1)
    result['nt_max_n'] = int(re.search(r'nt_max_n=(\d+)', string).group(1))
    result['t_max_n'] = int(re.search(r't_max_n=(\d+)', string).group(1))
    pca_match = re.search(r'pca\s*=\s*(\d+)', string)
    if pca_match:
        result['pca'] = int(pca_match.group(1))
    else:
        result['pca'] = None
    result['DCF_min'] = float(re.search(r':\s*([\d.]+)', string).group(1))
    results.append(result)
    

    



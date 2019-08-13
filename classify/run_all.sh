nohup python3 -u classify.py -train_file ../all_util -test_file none -fold -nb_classes 2 -emb data/embeddings/pt_word2vec_sg_600.emb -classifier linearsvm -balance > ../results/all_util_600SVM.res 
nohup python3 -u classify.py -train_file ../all_util -test_file none -fold -nb_classes 2 -emb data/embeddings/pt_word2vec_sg_600.emb -classifier mlp -balance > ../results/all_util_600MLP.res 
nohup python3 -u classify.py -train_file ../all_util -test_file none -fold -nb_classes 2 -emb data/embeddings/pt_word2vec_sg_600.emb -classifier randfor -balance > ../results/all_util_600RF.res &
nohup python3 -u classify.py -train_file ../all_sent -test_file none -fold -nb_classes 2 -emb data/embeddings/pt_word2vec_sg_600.emb -classifier randfor -balance > ../results/all_sent_600RF.res 

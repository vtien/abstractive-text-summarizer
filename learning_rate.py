import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train(model, num_epochs=50, lr=0.0003, print_every=100):
    """Train a model on IWSLT"""
    
    if USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    #criterion = LabelSmoothing(size=len(TRG.vocab), padding_idx=PAD_INDEX, smoothing=0.1)
    criterion.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler=StepLR(optim, step_size=2,gamma=0.1)
    
    dev_perplexities = []
    lr_list=[]

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optim),
                                     print_every=print_every)
        
        model.eval()
        with torch.no_grad():
            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
                           model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)        

            dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
                                       model, 
                                       SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
        lr_list.append(scheduler.get_lr())
        scheduler.step()

    return dev_perplexities, lr_list
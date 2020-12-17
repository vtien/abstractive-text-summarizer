
def beam_search_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None, beam_width=3):
    """Beam search to decode the summaries."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    sequences = [[list(), 0.0]]
    attention_scores = []
    hidden = None

    for i in range(max_len):
      all_candidates = []
      if i==0: #on first iteration, we only use the sos_index to feed into the decoder
          with torch.no_grad():
              out, hidden, pre_output = model.decode(encoder_hidden, encoder_final, src_mask,
                                                    prev_y, trg_mask, hidden)
              prob = model.generator(pre_output[:, -1])
              top_hiddens = [hidden]*beam_width #these will all be the same for first iteration

          attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
          seq, score = sequences[i]
          scores, next_words = torch.topk(prob, k=beam_width, dim=1) #scores are the log(prob) values and next_words are the indices
          scores, next_words = scores.data[0], next_words.data[0]
          for j in range(len(next_words)):
            candidates = [seq + [next_words[j].cpu().item()], score + scores[j].item()]
            all_candidates.append(candidates) #sequences are length 3 at the end of this

          ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
          sequences = ordered[:beam_width]

      else: #after first iteration, we have k words to use as prior distribution in order to decode next word
        hidden_states = []
        for k in range(len(sequences)):          
            with torch.no_grad():
                use_hidden = top_hiddens[k]
                prev_y = torch.ones(1, 1).type_as(src).fill_(next_words[k].item())
                out, hidden, pre_output = model.decode(encoder_hidden, encoder_final, src_mask,
                                                      prev_y, trg_mask, use_hidden)   
                prob = model.generator(pre_output[:, -1])
                hidden_states.append(hidden)

            attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
            seq, score = sequences[k]
            new_scores, new_next_words = torch.topk(prob, k=beam_width, dim=1) #next words are the indices
            new_scores, new_next_words = new_scores.data[0], new_next_words.data[0] #index into double list
            for j in range(len(new_scores)):
                candidates = [seq + [new_next_words[j].cpu().item()], score + new_scores[j].item()]
                all_candidates.append(candidates)

        #prune ouputs to include top k probabilities
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        sequences = ordered[:beam_width]

        #create hidden states to use in next time step based off the word use to achieve the best next word (likelihood, which is the second
        # to last number in the sequences structure)
        top_hiddens = []
        for h in range(len(next_words)):
          idx = next_words.tolist().index(sequences[h][0][-2])
          top_hiddens.append(hidden_states[idx])

        #create next words to be used as prev_y in decoder 
        next_words = []
        for w in range(len(sequences)):
          next_words.append(sequences[w][0][-1])
        next_words = torch.Tensor(next_words)
    
    #at end of loop prune to the single best sequence found through beam search. This will be the first sequence since we have been ordering them
    output = np.array(sequences[0][0])
        
    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
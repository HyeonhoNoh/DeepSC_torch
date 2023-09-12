from utils.util_seq2seq import *
from .model import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from .dataloader import Dataset
from tqdm import tqdm

def test(config):
    torch.manual_seed(0)

    trainer = SemanticTrainer(config)
    similarity = trainer.cal_similarity(1000)

    print((f"Similarity: {similarity:.3f}"))

def train(config):
    torch.manual_seed(0)

    trainer = SemanticTrainer(config)
    trainer.train()

class SemanticTrainer():
    def __init__(self, config):
        self.tokenizer = tokenizer(config)

        train_iter = Dataset(config, 'train')
        self.train_dataloader = DataLoader(train_iter, batch_size=config.param['batch_size'], collate_fn=self.tokenizer.collate_fn, num_workers=8)
        self.val_iter = Dataset(config, 'test')
        self.val_dataloader = DataLoader(self.val_iter, batch_size=config.param['batch_size'], collate_fn=self.tokenizer.collate_fn, num_workers=8)

        self.similarity_checker = Similarity()

        SRC_VOCAB_SIZE = len(self.tokenizer.vocab_transform)

        self.model = Transceiver(config, SRC_VOCAB_SIZE, SRC_VOCAB_SIZE)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model = self.model.to(config.device)

        self.config = config
        self.device = config.device
        self.set_loss()
        self.set_optimizer()

        if self.config.param['test']:
            self.load_network()

    def train(self):
        best_similarity = 0.
        for epoch in range(self.config.param['n_epochs']):
            self.model.train()
            losses = 0
            CE_losses = 0
            MSE_losses = 0

            pbar = tqdm(self.train_dataloader)

            for src in pbar:
                time.sleep(0.01)
                src = src.to(self.device)

                tgt_input = src[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

                outputs, channel_decoding, semantic_features = self.model(src,
                                                                          tgt_input,
                                                                          src_mask,
                                                                          tgt_mask,
                                                                          src_padding_mask,
                                                                          tgt_padding_mask,
                                                                          src_padding_mask,
                                                                          self.config.param['noise_var']
                                                                          )

                tgt_out = src[1:, :]

                CE_loss = torch.nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.shape[-1]),
                                                         tgt_out.reshape(-1))
                MSE_loss = torch.nn.MSELoss()(semantic_features, channel_decoding)
                loss = 0.95 * CE_loss + 0.05 * MSE_loss

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                losses += loss.item()
                CE_losses += CE_loss.item()
                MSE_losses += MSE_loss.item()

                torch.cuda.empty_cache()

            train_loss = losses / len(self.train_dataloader)
            CE_train_loss = CE_losses / len(self.train_dataloader)
            MSE_train_loss = MSE_losses / len(self.train_dataloader)

            val_loss = self.evaluate()
            similarity = self.cal_similarity(100)

            if similarity > best_similarity:
                best_similarity = similarity
                self.save_network()
                print("New record: ", best_similarity.item())

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train CE loss: {CE_train_loss:.3f}, Train MSE loss: {MSE_train_loss:.3f}"
                   f", Val loss: {val_loss:.3f}, Similarity: {similarity:.3f}, Best Similarity: {best_similarity:.3f}, "))

            print(self.translate("it is an important step towards equal rights for all passengers ."))
        self.save_network()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            losses = 0
            pbar = tqdm(self.val_dataloader)
            for src in pbar:
                src = src.to(self.device)

                tgt_input = src[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

                outputs, channel_decoding, semantic_features = self.model(src,
                                                                          tgt_input,
                                                                          src_mask,
                                                                          tgt_mask,
                                                                          src_padding_mask,
                                                                          tgt_padding_mask,
                                                                          src_padding_mask,
                                                                          self.config.param['test_noise_var']
                                                                          )

                tgt_out = src[1:, :]

                loss = 0.9 * self.CE_loss(outputs.reshape(-1, outputs.shape[-1]), tgt_out.reshape(-1)) + 0.1 * self.MSE_loss(
                    semantic_features, channel_decoding)
                losses += loss.item()

        return losses / len(self.val_dataloader)

    def cal_similarity(self, n_sim_data, coding_rate=None):
        similarity_list = []
        val_sen_list = self.val_iter.batch_sample(n_sim_data)
        for val_sen in val_sen_list:
            out_sen = self.translate(val_sen)
            sim = self.similarity_checker.compute_score(val_sen, out_sen)
            similarity_list.append(sim)

        return sum(similarity_list) / len(similarity_list)


    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        semantic_features = self.model.Semantic_Transformer.transformer_enc(src, src_mask, None)
        channel_encoding = self.model.Channel_Transformer.channel_enc(semantic_features)
        channel_encoding = self.model.channel(channel_encoding, self.config.param['test_noise_var'])
        channel_decoding = self.model.Channel_Transformer.channel_dec(channel_encoding)

        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len - 1):
            channel_decoding = channel_decoding.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), self.device)
                        .type(torch.bool)).to(self.device)
            out = self.model.Semantic_Transformer.transformer_dec.decode(ys, channel_decoding, tgt_mask, None, None)
            out = out.transpose(0, 1)
            prob = self.model.Semantic_Transformer.transformer_dec.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

    def translate(self, src_sentence: str):
        self.model.eval()
        src = self.tokenizer.text_transform()(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

        return " ".join(
            self.tokenizer.vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(
            "<bos>",
            "").replace(
            "<eos>", "")

    def set_loss(self):
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.MSE_loss = torch.nn.MSELoss()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def save_network(self):
        semantic_network_name = self.config.param['model_name'] + "_" + str(self.config.param['SNR'])
        torch.save(self.model.state_dict(),
                   self.config.paths['checkpoint_path'] + "/latest_%s.pth" % (semantic_network_name))

        print('Successfully save the latest model parameters.')

    def load_network(self):
        semantic_network_name = self.config.param['model_name'] + "_" + str(self.config.param['SNR'])
        self.model.load_state_dict(torch.load(self.config.paths['checkpoint_path'] + "/latest_%s.pth" % (semantic_network_name)))

        print('Successfully load the latest model parameters.')

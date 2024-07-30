from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
from einops import rearrange


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []



        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights



'''
def clip_classifier(classnames, template, clip_model):

    clip_weights = []



    for classname in classnames:
        # Tokenize the prompts
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        texts = clip.tokenize(texts).cuda()
        # prompt ensemble for ImageNet
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        clip_weights.append(class_embedding)

    clip_weights = torch.stack(clip_weights, dim=1).cuda()

    return clip_weights
'''










def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        cache_keys_obj = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
            #for augment_idx in range(1):


                train_features = []
                train_features_obj = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features, image_features_obj = clip_model.encode_image(images)
                    train_features.append(image_features)

                    train_features_obj.append(image_features_obj)



                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_keys_obj.append(torch.cat(train_features_obj, dim=0).unsqueeze(0))


        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        #cache_ori = torch.cat(cache_keys_obj, dim=0)
        #print(cache_keys.shape)

        cache_keys_obj = torch.cat(cache_keys_obj, dim=0).mean(dim=0)
        #print(cache_keys_obj.shape)
        cache_keys_obj /= cache_keys_obj.norm(dim=-1, keepdim=True)
        #cache_keys_obj = cache_keys_obj.permute(2, 0)
        cache_keys_obj = rearrange(cache_keys_obj, 'a1 a2 a3-> a3 a2 a1')

        #print(cache_keys_obj.shape)



        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()



        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

        torch.save(cache_keys_obj, cfg['cache_dir'] + '/keys_obj_' + str(cfg['shots']) + "shots.pt")


    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

        cache_keys_obj = torch.load(cfg['cache_dir'] + '/keys_obj_' + str(cfg['shots']) + "shots.pt")





    return cache_keys, cache_values, cache_keys_obj


def pre_load_features(cfg, split, clip_model, loader, prompt_cls, new_trans):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []
        features_obj = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features, image_features_obj = clip_model.encode_image(images, prompt_cls, new_trans)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)



                image_features_obj /= image_features_obj.norm(dim=-1, keepdim=True)
                features_obj.append(image_features_obj)


                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)



        features_obj = torch.cat(features_obj)



        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   


        torch.save(features_obj, cfg['cache_dir'] + "/" + split + "_f_o.pt")


    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")



        features_obj = torch.load(cfg['cache_dir'] + "/" + split + "_f_o.pt")
    


    return features, labels, features_obj


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None, adapter_obj = None, val_features_obj=None, cache_keys_obj=None):


    factor = cfg['shots']/16.0
    if factor > 0.25:
        factor = 0.25
    
    if cfg['shots'] == 16:
        factor = 0.5

        
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity,  image_features_new, clip_weights_new  = adapter(features, clip_weights)

                    affinity_obj,  image_features_new_obj, clip_weights_new_obj  = adapter_obj(val_features_obj, clip_weights)


                else:
                    affinity = features @ cache_keys
                    image_features_new = features
                    clip_weights_new = clip_weights


                    affinity_obj = val_features_obj @ cache_keys_obj
                    image_features_new_obj = val_features_obj
                    clip_weights_new_obj = clip_weights






                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * image_features_new @ clip_weights_new
                tip_logits1 = clip_logits + cache_logits * alpha

                cache_logits_obj = ((-1) * (beta - beta * affinity_obj)).exp() @ cache_values
                clip_logits_obj = 100. * image_features_new_obj @ clip_weights_new_obj
                tip_logits2 = clip_logits_obj + cache_logits_obj * alpha

                tip_logits = tip_logits1 + factor*tip_logits2

                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        #print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_acc))

        '''
        with open('./acc_record.txt', 'a') as file:
            file.write("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_acc))
        '''


    return best_beta, best_alpha

import time
import torch
import torch.optim as optim

from ProtoPNet.helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, get_full_results=False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None

    if is_train:
        mode = "train"
    else:
        mode = "test"

    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, batch in enumerate(dataloader):
        # Move each tensor in the batch to device
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'token_type_ids': batch['token_type_ids'].to(device),
            'visual_embeds': batch['visual_embeds'].to(device),
            'visual_attention_mask': batch['visual_attention_mask'].to(device)
        }
        target = batch['label'].to(device)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(inputs)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,target]).to(device)
                inverted_distances, _ = torch.max((1.0 - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(1.0 - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((1.0 - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(1.0 - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(device)
                    m = model.module if hasattr(model, 'module') else model
                    l1 = (m.last_layer.weight * l1_mask).norm(p=1)
                else:
                    m = model.module if hasattr(model, 'module') else model
                    l1 = m.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                m = model.module if hasattr(model, 'module') else model
                l1 = m.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del inputs
        del target
        del output
        del predicted
        del min_distances

        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    end = time.time()

    # save the results
    running_time = end - start
    cross_entropy = total_cross_entropy / n_batches
    cluster = total_cluster_cost / n_batches
    if class_specific:
        separation = total_separation_cost / n_batches
        avg_separation = total_avg_separation_cost / n_batches
    else:
        separation = 0
        avg_separation = 0

    accu = n_correct / n_examples * 100
    l1 = model.module.last_layer.weight.norm(p=1).item()

    log('\ttime: \t{0}'.format(running_time))
    log('\tcross ent: \t{0}'.format(cross_entropy))
    log('\tcluster: \t{0}'.format(cluster))
    if class_specific:
        log('\tseparation:\t{0}'.format(separation))
        log('\tavg separation:\t{0}'.format(avg_separation))
    log('\taccu: \t\t{0}%'.format(accu))
    log('\tl1: \t\t{0}'.format(l1))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    if get_full_results:
        return mode, running_time, cross_entropy, cluster, separation, avg_separation, accu, l1, p_avg_pair_dist
    else:
        return accu


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, get_full_results=False):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, get_full_results=get_full_results)


def validate(model, dataloader, class_specific=False, log=print, get_full_results=False):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, get_full_results=get_full_results)



def warm_only_multimodal(model, log=print):
    """Freeze pretrained layers, train only prototype-related parts"""
    # Freeze pretrained feature extractors
    for p in model.module.vgg16_features.parameters():
        p.requires_grad = False
    for p in model.module.encoder.parameters():
        p.requires_grad = False
    
    # Enable training for prototype-related layers
    for p in model.module.visual_projection.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    
    log('\twarm (multimodal)')


def joint_multimodal(model, log=print):
    """Enable training for all layers"""
    # Enable all layers
    for p in model.module.vgg16_features.parameters():
        p.requires_grad = True
    for p in model.module.encoder.parameters():
        p.requires_grad = True
    for p in model.module.visual_projection.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    
    log('\tjoint (multimodal)')


def last_only_multimodal(model, log=print):
    """Train only the last layer"""
    # Freeze everything except last layer
    for p in model.module.vgg16_features.parameters():
        p.requires_grad = False
    for p in model.module.encoder.parameters():
        p.requires_grad = False
    for p in model.module.visual_projection.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer (multimodal)')


import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import json
import numpy as np 
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def chat_template(example):
    prompt_type = "vanilla"
    print(example[prompt_type])
    conversation = [
        {"role": "user", "content": example[prompt_type]}
    ]
    return {"conversation": conversation}

def get_activations(model, instructions, tokenize_instructions_fn, batch_size=16, layer_idx=-1, token_idx=-1):
    last_activations= None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with torch.no_grad(): 
            outputs = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                output_hidden_states=True
            )

        activations = outputs.hidden_states[layer_idx][:, token_idx, :] 

        if last_activations is None:
            last_activations = activations
        else:
            last_activations = torch.cat((last_activations, activations), dim=0)
    return last_activations


def compute_and_plot_reduction_with_classifier(activation_refusal_lg1, activation_nn_refusal_lg1, activation_harmful_lg2, activation_harmless_lg2, label_data_lg2_refusal,  dir, lg2, checkpoint, reduction_type='pca', n_compo=2, prompt_type="vanilla", classifier="svm", classifier_form_prompts=False, layer=-1): 
    activations_ref_lg1_np = activation_refusal_lg1.detach().cpu().to(torch.float32) 
    activations_nn_ref_lg1_np = activation_nn_refusal_lg1.detach().cpu().to(torch.float32)

    activations_harmful_np= activation_harmful_lg2.detach().cpu().to(torch.float32)
    activations_harmless_np = activation_harmless_lg2.detach().cpu().to(torch.float32)

    X_comp_lg1 = torch.cat([activations_ref_lg1_np, activations_nn_ref_lg1_np], dim=0).numpy()
    X_rep = torch.cat([activations_harmful_np, activations_harmless_np], dim=0).numpy()

    labels_rep = ["Harmful"] * activations_harmful_np.shape[0] + ["Harmless"] * activations_harmless_np.shape[0]
    labels_binary = [1] * activations_harmful_np.shape[0] + [0] * activations_harmless_np.shape[0]
    labels_type = ["Vanilla"] * 150 + ["Adversarial"] * 150  + ["Vanilla"] * 150 + ["Adversarial"] * 150 
    labels_refusal = ["Refusal" if x==1 else "Non refusal" for x in label_data_lg2_refusal]

    # Reduction computed on Refusal vs non Refusal in lg1 
    # Projection on Harmful vs Harmless in lg2 
    if reduction_type=='pca': 
        print("Compute PCA")
        pca = PCA(n_components=n_compo, random_state=42)
        X_comp_embedded_lg1 = pca.fit(X_comp_lg1)
        X_rep_embedded = pca.transform(X_rep)
    else :  
        raise NotImplementedError
    
    scaler = StandardScaler()
    X_2d = X_rep_embedded[:, :2]
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Train linear classifier on Harmful vs Harmless in lg2 
    if classifier=="svm":
        classifier_model = SVC(kernel='linear', probability=True, random_state=42)
    elif classifier=="regression":
        classifier_model = LogisticRegression(random_state=42, max_iter=1000)
    else :  
        raise NotImplementedError
    print("Train classifier")
    classifier_model.fit(X_2d_scaled, labels_binary)

    # Plot decision boundary on mesh grid  
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                        np.arange(y_min, y_max, 1))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = classifier_model.decision_function(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(15, 6))
    ax_plot = fig.add_subplot(121)
    ax_plot.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='coolwarm')
    ax_plot.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2, label=f'{classifier} boundary harmful vs harmless')

    if classifier_form_prompts==True:

        # Classification Vanilla vs Adversarial
        if classifier=="svm":
            classifier_model_form = SVC(kernel='linear', probability=True, random_state=42)
        elif classifier=="regression":
            classifier_model_form = LogisticRegression(random_state=42, max_iter=1000)
        else :  
            raise NotImplementedError
        classifier_model_form.fit(X_2d_scaled, labels_type)
        Z_form = classifier_model_form.decision_function(mesh_points_scaled)
        Z_form = Z_form.reshape(xx.shape)
        ax_plot.contour(xx, yy, Z_form, levels=[0], colors='black', linestyles='dotted', linewidths=2, label=f'{classifier} boundary adversarial vs vanilla')

        # Classificaiton Refusal vs nn refusal 
        if classifier=="svm":
            classifier_model_refusal = SVC(kernel='linear', probability=True, random_state=42)
        elif classifier=="regression":
            classifier_model_refusal = LogisticRegression(random_state=42, max_iter=1000)
        else :  
            raise NotImplementedError
        classifier_model_refusal.fit(X_2d_scaled, labels_refusal)
        Z_refusal= classifier_model_refusal.decision_function(mesh_points_scaled)
        Z_refusal = Z_refusal.reshape(xx.shape)

        ax_plot.contour(xx, yy, Z_refusal, levels=[0], colors='red', linestyles='dotted', linewidths=2, label=f'{classifier} boundary refusal vs non refusal')

    # Plot 
    colors = {"Harmful": 'red', "Harmless": 'blue'}
    if prompt_type=="all": 
        markers={"Vanilla":"o", "Adversarial":"*"}
        edge_colors={"Refusal":"green", "Non refusal":None}
        print("all")
        for i, (point, label, label_t, label_ref) in enumerate(zip(X_rep_embedded, labels_rep, labels_type, labels_refusal )):
            ax_plot.scatter(point[0], point[1], color=colors[label], marker=markers[label_t], edgecolors=edge_colors[label_ref], alpha=0.7, label=f"{label}-{label_t}")
        
        type_handles = []
        for c in markers.values() :
            type_handles.append(plt.scatter([], [], c="black", 
                                        marker=c, s=40, alpha=0.8, 
                                        edgecolors='black', linewidth=0.3))
        
    else : 
                
        for point, label in zip(X_rep_embedded, labels_rep):
            ax_plot.scatter(point[0], point[1], color=colors[label],  alpha=0.7, label=f"{label}")

    
    lang_handles = []
    for c in colors.values() :
        lang_handles.append(plt.scatter([], [], c=c, 
                                    marker='o', s=40, alpha=0.8, 
                                    edgecolors='black', linewidth=0.3))

    # Legends 
    clean_handles = []
    clean_labels = []
    for label, color in colors.items():
        clean_handles.append(plt.scatter([], [], c=color, marker='o', s=40, alpha=0.8, edgecolors='none'))
        clean_labels.append(label)
    for label, marker in markers.items():
        clean_handles.append(plt.scatter([], [], c="gray", marker=marker, s=40, alpha=0.8, edgecolors='none'))
        clean_labels.append(f"Type: {label}")
    for label, edge_color in edge_colors.items():
        clean_handles.append(plt.scatter([], [], c="white", marker='o', s=40, alpha=0.8, 
                                    edgecolors=edge_color, linewidth=2))
        clean_labels.append(f"Refusal: {label}")


    # Make prediction and store in json files 
    results = {}
    results_all = {}
    results_vanilla = {}
    results_adversarial= {}
    
    # Prediction Harmful vs Harmless (all forms) 
    y_proba_all_harmfulness = classifier_model.predict_proba(X_2d_scaled)[:, 1]
    accuracy_all_harmfulness = classifier_model.score(X_2d_scaled, labels_binary)
    auc_score_all_harmfulness = roc_auc_score(labels_binary, y_proba_all_harmfulness)
    results_all["harmfulness"]={"accuracy":accuracy_all_harmfulness, 
                                "auc":auc_score_all_harmfulness}

    metrics_text_harmful_harmless = f"{classifier} Harmful vs Harmless:\nAccuracy: {accuracy_all_harmfulness:.3f}\nAUC: {auc_score_all_harmfulness:.3f}"

    # Prediction Vanilla vs Adversarial (all forms) 
    y_proba_all_form = classifier_model_form.predict_proba(X_2d_scaled)[:, 1]
    accuracy_all_form = classifier_model_form.score(X_2d_scaled, labels_type)
    auc_score_all_form = roc_auc_score(labels_type, y_proba_all_form)
    results_all["adversarial_vs_vanilla"]={"accuracy":accuracy_all_form, 
                            "auc":auc_score_all_form}
    metrics_text_vanilla_adv = f"{classifier} Vanilla vs Adversarial:\nAccuracy: {accuracy_all_form:.3f}\nAUC: {auc_score_all_form:.3f}"

    # Prediction Refusal vs Non Refusal (all forms) 
    y_proba_all_refusal = classifier_model_refusal.predict_proba(X_2d_scaled)[:, 1]
    accuracy_all_refusal = classifier_model_refusal.score(X_2d_scaled, labels_refusal)
    auc_score_all_refusal = roc_auc_score(labels_refusal, y_proba_all_refusal)
    results_all["refusal"]={"accuracy":accuracy_all_refusal, 
                            "auc":auc_score_all_refusal}
    metrics_text_refusal = f"{classifier} Refusal vs Non refusal:\nAccuracy: {accuracy_all_refusal:.3f}\nAUC: {auc_score_all_refusal:.3f}"

    # Add metrics in legend
    legend_labels = clean_labels + [metrics_text_harmful_harmless, metrics_text_vanilla_adv, metrics_text_refusal]
    legend_handles = clean_handles + [
        plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='--'),
        plt.Line2D([0], [0], color='black', alpha=0.7, linestyle='dotted'),
        plt.Line2D([0], [0], color='red', alpha=0.7, linestyle='dotted')
    ]
    legend = ax_plot.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())

    table_left = legend_bbox_fig.x0
    table_bottom = legend_bbox_fig.y0 - 0.25  
    table_width = legend_bbox_fig.width
    table_height = 0.15

    ax_table = fig.add_axes([table_left, table_bottom, table_width, table_height])
    ax_table.axis('off')

    ax_plot.set_title(f"Latent space {reduction_type} of model activations Harmful vs Harmless with Refusal in {lg2} - {checkpoint}")
    ax_plot.set_xlabel("Dim 1")
    ax_plot.set_ylabel("Dim 2")
    ax_plot.grid(True)

    if classifier_form_prompts==True: 
        name_fig = f"{dir}/latent_space_{reduction_type}_{lg2}_{classifier}_with_classification_layer_{layer}.png"
    else : 
        name_fig = f"{dir}/latent_space_{reduction_type}_{lg2}_{classifier}.png"
    plt.subplots_adjust(right=0.7)  
    plt.tight_layout()
    plt.savefig(name_fig)
    plt.show()

    # Make predictions for each class (vanilla and adversarial)
    # Train classifier for each class 
    X_vanilla = np.concatenate([X_rep_embedded[:150], X_rep_embedded[300:450]], axis=0)
    X_adv = np.concatenate([X_rep_embedded[150:300], X_rep_embedded[450:]], axis=0)

    X_2d_vanilla = X_vanilla[:, :2]
    X_2d_adv = X_adv[:, :2]

    scaler = StandardScaler()
    X_2d_vanilla_scaled = scaler.fit_transform(X_2d_vanilla)
    X_2d_adv_scaled = scaler.fit_transform(X_2d_adv)
    refusal_label_data_list = label_data_lg2_refusal.tolist()
   
    labels_harmfulness = ["Harmful"] * 150 + ["Harmless"] * 150
    labels_refusal_vanilla = ["Refusal" if x==1 else "Non refusal" for x in refusal_label_data_list[:150]+refusal_label_data_list[300:450]]
    labels_refusal_adv = ["Refusal" if x==1 else "Non refusal" for x in refusal_label_data_list[150:300]+refusal_label_data_list[450:]]


    if classifier=="svm":
        classifier_vanilla_harmfulness = SVC(kernel='linear', probability=True, random_state=42)
        classifier_vanilla_refusal = SVC(kernel='linear', probability=True, random_state=42)
        classifier_adv_harmfulness = SVC(kernel='linear', probability=True, random_state=42)
        classifier_adv_refusal = SVC(kernel='linear', probability=True, random_state=42)
    elif classifier=="regression":
        classifier_vanilla_harmfulness = LogisticRegression(random_state=42, max_iter=1000)
        classifier_vanilla_refusal = LogisticRegression(random_state=42, max_iter=1000)
        classifier_adv_harmfulness = LogisticRegression(random_state=42, max_iter=1000)
        classifier_adv_refusal = LogisticRegression(random_state=42, max_iter=1000)

    classifier_vanilla_harmfulness.fit(X_2d_vanilla_scaled, labels_harmfulness)
    classifier_vanilla_refusal.fit(X_2d_vanilla_scaled, labels_refusal_vanilla)
    classifier_adv_harmfulness.fit(X_2d_adv_scaled, labels_harmfulness)
    classifier_adv_refusal.fit(X_2d_adv_scaled, labels_refusal_adv)

    # Prediction for each class (vanilla and adversarial)
    # Harmful vs Harmless vanilla 
    y_proba_vanilla_harmfulness = classifier_vanilla_harmfulness.predict_proba(X_2d_vanilla_scaled)[:, 1]
    accuracy_vanilla_harmfulness= classifier_vanilla_harmfulness.score(X_2d_vanilla_scaled, labels_harmfulness)
    auc_score_vanilla_harmfulness = roc_auc_score(labels_harmfulness, y_proba_vanilla_harmfulness)

    # Harmful vs Harmless adversarial 
    y_proba_adv_harmfulness = classifier_adv_harmfulness.predict_proba(X_2d_adv_scaled)[:, 1]
    accuracy_adv_harmfulness = classifier_adv_harmfulness.score(X_2d_adv_scaled, labels_harmfulness)
    auc_score_adv_harmfulness = roc_auc_score(labels_harmfulness, y_proba_adv_harmfulness)

    # Refusal vs Non Refusal vanilla 
    y_proba_refusal_vanilla_refusal = classifier_vanilla_refusal.predict_proba(X_2d_vanilla_scaled)[:, 1]
    accuracy_vanilla_refusal = classifier_vanilla_refusal.score(X_2d_vanilla_scaled, labels_refusal_vanilla)
    auc_score_vanilla_refusal = roc_auc_score(labels_refusal_vanilla, y_proba_refusal_vanilla_refusal)

    # Refusal vs Non Refusal adversarial 
    y_proba_refusal_adv_refusal = classifier_adv_refusal.predict_proba(X_2d_adv_scaled)[:, 1]
    accuracy_adv_refusal = classifier_adv_refusal.score(X_2d_adv_scaled, labels_refusal_adv)
    auc_score_adv_refusal = roc_auc_score(labels_refusal_adv, y_proba_refusal_adv_refusal)

    results_vanilla["harmfulness"]={"accuracy":accuracy_vanilla_harmfulness, 
                            "auc":auc_score_vanilla_harmfulness}
    results_adversarial["harmfulness"]={"accuracy":accuracy_adv_harmfulness, 
                            "auc":auc_score_adv_harmfulness}
    results_vanilla["refusal"]={"accuracy":accuracy_vanilla_refusal, 
                            "auc":auc_score_vanilla_refusal}
    results_adversarial["refusal"]={"accuracy":accuracy_adv_refusal, 
                            "auc":auc_score_adv_refusal}
    
    results = {"all": results_all, 
               "vanilla": results_vanilla, 
               "adversarial": results_adversarial}
    
    path_json = f"{dir}/results_{reduction_type}_{lg2}_{classifier}_with_classification_layer_{layer}.json"
    with open(path_json, 'w') as f:
        json.dump(results, f, indent=4)

    return X_rep_embedded
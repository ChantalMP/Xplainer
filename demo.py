from pathlib import Path

import gradio as gr
import numpy as np
from matplotlib import pyplot as plt

from descriptors import disease_descriptors_chexpert, disease_descriptors_chestxray14
from model import InferenceModel


def plot_bars(model_output):
    # Create a figure with as many subplots as there are diseases, arranged vertically
    fig, axs = plt.subplots(len(model_output), 1, figsize=(10, 5 * len(model_output)))
    # axs is not iterable if only one subplot is created, so make it a list
    if len(model_output) == 1:
        axs = [axs]

    for ax, (disease, data) in zip(axs, model_output.items()):
        desc_probs = list(data['descriptor_probabilities'].items())

        my_probs = [p[1] for p in desc_probs]
        min_prob = min(my_probs)
        max_prob = max(my_probs)
        my_labels = [p[0] for p in desc_probs]

        # Convert probabilities to differences from 0.5
        diffs = np.abs(np.array(my_probs) - 0.5)

        # Set colors based on sign of difference
        colors = ['red' if p < 0.5 else 'forestgreen' for p in my_probs]

        # Plot bars with appropriate colors and left offsets
        left = [p if p < 0.5 else 0.5 for p in my_probs]
        bars = ax.barh(my_labels, diffs, left=left, color=colors, alpha=0.3)

        for i, bar in enumerate(bars):
            ax.text(min_prob - 0.04, bar.get_y() + bar.get_height() / 2, my_labels[i], ha='left', va='center', color='black', fontsize=15)

        ax.set_xlim(min(min_prob - 0.05, 0.49), max(max_prob + 0.05, 0.51))

        # Invert the y-axis to show bars with values less than 0.5 to the left of the center
        ax.invert_yaxis()

        ax.set_yticks([])

        # Add a title for the disease
        ax.set_title(f"Probability of {disease} : {data['overall_probability']:.2f}")
        # make title larger and bold
        ax.title.set_fontsize(15)
        ax.title.set_fontweight(600)

    # Save the plot
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    file_path = 'plot.png'
    plt.savefig(file_path)
    plt.close(fig)

    return file_path


def classify_image(inference_model, image_path, diseases_to_predict):
    descriptors_with_indication = [d + " indicating " + disease for disease, descriptors in diseases_to_predict.items() for d in descriptors]
    probs, negative_probs = inference_model.get_descriptor_probs(image_path=Path(image_path), descriptors=descriptors_with_indication,
                                                                 do_negative_prompting=True, demo=True)

    disease_probs, negative_disease_probs = inference_model.get_diseases_probs(diseases_to_predict, pos_probs=probs, negative_probs=negative_probs)

    model_output = {}
    for idx, disease in enumerate(diseases_to_predict.keys()):
        model_output[disease] = {
            'overall_probability': disease_probs[disease],
            'descriptor_probabilities': {descriptor: probs[f'{descriptor} indicating {disease}'].item() for descriptor in
                                         diseases_to_predict[disease]}
        }

    file_path = plot_bars(model_output)
    return file_path


# Define the function you want to wrap
def process_input(image_path, prompt_names: list, disease_name: str, descriptors: str):
    diseases_to_predict = {}

    for prompt in prompt_names:
        if prompt == 'Custom':
            diseases_to_predict[disease_name] = descriptors.split('\n')
        else:
            if prompt in disease_descriptors_chexpert:
                diseases_to_predict[prompt] = disease_descriptors_chexpert[prompt]
            else:  # only chestxray14
                diseases_to_predict[prompt] = disease_descriptors_chestxray14[prompt]

    # classify
    model = InferenceModel()
    output = classify_image(model, image_path, diseases_to_predict)

    return output


if __name__ == '__main__':
    # Define the Gradio interface
    iface = gr.Interface(
        fn=process_input,
        inputs=[gr.inputs.Image(type="filepath"), gr.inputs.CheckboxGroup(
            choices=['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                     'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices',
                     'Infiltration', 'Mass', 'Nodule', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
                     'Custom'],
            default=['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                     'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'],
            label='Selct to use predefined disease descriptors. Select "Custom" to define your own observations.'),
                gr.inputs.Textbox(lines=2, placeholder="Name of pathology for which you want to define custom observations", label='Pathology:'),
                gr.inputs.Textbox(lines=2, placeholder="Add your custom (positive) observations separated by a new line"
                                                       "\n Note: Each descriptor will automatically be embedded into our prompt format: There is/are (no) <observation> indicating <pathology>"
                                                       "\n Example:\n\n Opacity\nPleural Effusion\nConsolidation"
                                  , label='Custom Observations:')],
        outputs=gr.outputs.Image(type="filepath")
    )

    # Launch the interface
    iface.launch(share=True)

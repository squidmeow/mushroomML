import gradio as gr
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load model
pipeline = joblib.load("mushroom_pipeline.pkl")

encoder = pipeline.named_steps["encoder"]
model = pipeline.named_steps["classifier"]

def predict_edibility(odor, cap_shape, cap_color, gill_size, gill_color, habitat, bruises):
    data_dict2 = {
    'cap-shape':{'bell':'b','conical':'c','convex':'x','flat':'f','knobbed':'k','sunken':'s'},
    'cap-color':{'brown':'n','buff':'b','cinnamon':'c','gray':'g','green':'r','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'},
    'bruises':{'bruises':'t','no':'f'},
    'odor':{'almond':'a','anise':'l','creosote':'c','fishy':'y','foul':'f','musty':'m','none':'n','pungent':'p','spicy':'s'},
    'gill-size':{'broad':'b','narrow':'n'},
    'gill-color':{'black':'k','brown':'n','buff':'b','chocolate':'h','gray':'g','green':'r','orange':'o','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'},
    'habitat':{'grasses':'g','leaves':'l','meadows':'m','paths':'p','urban':'u','waste':'w','woods':'d'}
}
    input_data = pd.DataFrame([{
        'odor': data_dict2["odor"][odor],
        'cap-shape': data_dict2["cap-shape"][cap_shape],
        'cap-color': data_dict2["cap-color"][cap_color],
        'gill-size': data_dict2["gill-size"][gill_size],
        'gill-color': data_dict2["gill-color"][gill_color],
        'habitat': data_dict2["habitat"][habitat],
        'bruises': data_dict2["bruises"][bruises]

    }])
    
    input_encoded = encoder.transform(input_data)
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.feature_names_in_)
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded).max()
    label = f"Edible 🍽️" if pred == 1 else "Poisonous ☠️"
    label += f" (Confidence: {prob:.2f})"

    if pred == 1 and prob < 0.8:
      label += f" ⚠️ Low confidence—please verify!"

    # Confidence gauge
    fig1, ax1 = plt.subplots(figsize=(4, 1.5))
    ax1.barh(['Confidence'], [prob], color='green' if prob > 0.8 else 'orange')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Confidence Score', fontsize=8)
    ax1.set_title('Confidence Gauge', fontsize=10)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Feature importance chart
    importances = model.feature_importances_
    features = encoder.feature_names_in_
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(features, importances, color='skyblue')
    ax2.set_title('Feature Importance')

    return fig1, fig2, gr.update(value=f"# {label}", visible=True), gr.update(visible=True)

data_dict = {
    'class':{'edible':'e','poisonous':'p'},
    'cap-shape':{'bell':'b','conical':'c','convex':'x','flat':'f','knobbed':'k','sunken':'s'},
    'cap-surface':{'fibrous':'f','grooves':'g','scaly':'y','smooth':'s'},
    'cap-color':{'brown':'n','buff':'b','cinnamon':'c','gray':'g','green':'r','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'},
    'bruises':{'bruises':'t','no':'f'},
    'odor':{'almond':'a','anise':'l','creosote':'c','fishy':'y','foul':'f','musty':'m','none':'n','pungent':'p','spicy':'s'},
    'gill-attachment':{'attached':'a','descending':'d','free':'f','notched':'n'},
    'gill-spacing':{'close':'c','crowded':'w','distant':'d'},
    'gill-size':{'broad':'b','narrow':'n'},
    'gill-color':{'black':'k','brown':'n','buff':'b','chocolate':'h','gray':'g','green':'r','orange':'o','pink':'p','purple':'u','red':'e','white':'w','yellow':'y'},
    'stalk-shape':{'enlarging':'e','tapering':'t'},
    'stalk-root':{'bulbous':'b','club':'c','cup':'u','equal':'e','rhizomorphs':'z','rooted':'r','missing':'?'},
    'stalk-surface-above-ring':{'fibrous':'f','scaly':'y','silky':'k','smooth':'s'},
    'stalk-surface-below-ring':{'fibrous':'f','scaly':'y','silky':'k','smooth':'s'},
    'stalk-color-above-ring':{'brown':'n','buff':'b','cinnamon':'c','gray':'g','orange':'o','pink':'p','red':'e','white':'w','yellow':'y'},
    'stalk-color-below-ring':{'brown':'n','buff':'b','cinnamon':'c','gray':'g','orange':'o','pink':'p','red':'e','white':'w','yellow':'y'},
    'veil-type':{'partial':'p','universal':'u'},
    'veil-color':{'brown':'n','orange':'o','white':'w','yellow':'y'},
    'ring-number':{'none':'n','one':'o','two':'t'},
    'ring-type':{'cobwebby':'c','evanescent':'e','flaring':'f','large':'l','none':'n','pendant':'p','sheathing':'s','zone':'z'},
    'spore-print-color':{'black':'k','brown':'n','buff':'b','chocolate':'h','green':'r','orange':'o','purple':'u','white':'w','yellow':'y'},
    'population':{'abundant':'a','clustered':'c','numerous':'n','scattered':'s','several':'v','solitary':'y'},
    'habitat':{'grasses':'g','leaves':'l','meadows':'m','paths':'p','urban':'u','waste':'w','woods':'d'}
}

with gr.Blocks() as demo:
    gr.Markdown("# 🍄 Fungi or Faux Pas?")
    gr.Markdown("### Mushroom Edibility Predictor")
    gr.Markdown("Select mushroom features to predict if it's edible or poisonous. Confidence score included!")

    with gr.Row():
        with gr.Column():
            odor = gr.Dropdown(list(data_dict["odor"].keys()), label="Odor")
            cap_shape = gr.Dropdown(list(data_dict["cap-shape"].keys()), label="Cap Shape")
            cap_color = gr.Dropdown(list(data_dict["cap-color"].keys()), label="Cap Color")
            gill_size = gr.Dropdown(list(data_dict["gill-size"].keys()), label="Gill Size")
        with gr.Column():
            gill_color = gr.Dropdown(list(data_dict["gill-color"].keys()), label="Gill Color")
            habitat = gr.Dropdown(list(data_dict["habitat"].keys()), label="Habitat")
            bruises = gr.Dropdown(list(data_dict["bruises"].keys()), label="Bruises")

    submit = gr.Button("Predict")
    
    with gr.Row():
        with gr.Column(): 
            prediction_output = gr.Markdown("## Prediction will appear here", visible=False)
        with gr.Column():
            confidence_plot = gr.Plot(visible=False)

    #with gr.Row(): output = gr.Textbox(label="Prediction")
    #with gr.Row(): 
    with gr.Row(): importance_plot = gr.Plot(label="Feature Importance")
    
    submit.click(fn=predict_edibility, inputs=[odor, cap_shape, cap_color, gill_size, gill_color, habitat, bruises], outputs=[confidence_plot, importance_plot, prediction_output, confidence_plot])

    #submit.click(fn=predict_edibility, inputs=[odor, cap_shape, cap_color, gill_size, gill_color, habitat, bruises], outputs=output)

demo.launch()
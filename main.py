import gradio as gr
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import json

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set up the API key and project ID for IBM Watson 
watsonx_API = config['watsonx_API']
project_id = config['project_id']
model_idx = config['model_id']
resource_location_url = config['resource_location_url']

generate_params = {
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.TEMPERATURE: 0.7,
}

model = Model(
    model_id=model_idx,
    params=generate_params,
    credentials={
        "apikey": watsonx_API,
        "url": resource_location_url
    },
    project_id=project_id
)

def generate_proposal(client_requirements):
    try:
        prompt = f"Create a professional Upwork proposal based on the following client requirements and generate only the proposal:\n\n{client_requirements}"
        generated_response = model.generate(prompt=prompt)
        return generated_response['results'][0]['generated_text']
    except Exception as e:
        return str(e)

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_proposal,
    inputs=gr.Textbox(lines=10, placeholder="Paste the client's requirements here...", label="Client Requirements"),
    outputs=gr.Textbox(lines=20, label="Generated Proposal", interactive=True),
    title="Upwork Proposal Generator",
    description="Generate a professional Upwork proposal based on client requirements. Paste the requirements and get a well-crafted proposal."
)

# Launch the Gradio interface
iface.launch()

{% set models_to_generate = codegen.get_models(directory='core', prefix='') %}
{{ codegen.generate_model_yaml(
    model_names = models_to_generate
) }}
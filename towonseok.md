# Run file
```angular2html
python ppfl_ex.py    
```

# source code
/ppfl/personalization
network.py : network 생성 / 관리
learn.py: learning procedure
personalize layer: custom layer 

personalize layer --> network

## config file
/config/cfg.yaml


### network 구조
network.input_size
network.n_layers
network.n_hidden_units

### Vertical net 구조
personalized.vertical

### personalized net 구조
personalized.network

* 유의: network.n_layers == personalized.vertical.n_layers == personalized.network.n_layers

### train 구조
personalized.vertica.train ~


~~~
from ppfl.personalization.network import PersonalizedProgressiveNetwork
from ppfl.personalization.learn import PersonalizedNetworkLearn

[comment]: <> (common_model: models.Model)

ppfl = PersonalizedProgressiveNetwork(
    config, PersonalizedNetworkLearn, common_model
)

[comment]: <> (ppfl.network) : final model

ppfl.learn(common_inputs, vertical_inputs, labels, valid_data=None, vertical=True)
valid_data: tuple or list , valid_data[0]: inputs, valid_data[1]: labels
~~~
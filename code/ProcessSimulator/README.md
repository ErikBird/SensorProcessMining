## 

### Data Format of attributes
All Attributes need to be compatible to float. Boolean and String are not.

One Hot in Generator.

Padding of the first events.

### Create a Model
### Inspect the Model

See the DOT Representation of the Graph:
```python
print(model)
```
Plot the Graph:
```python
# if you have enabled widgetsnbextension for jupyter:
#model.print_graph_interactive()
#else
model.print_graph()
```


#model.save_graph_png()




dataset = tf.data.Dataset.from_generator(
    model.generator,
    output_signature=model.get_output_signature())

#%%

model.get_output_signature()


#%%

for input_example, target_example in dataset.take(1):
    print("Input :", input_example)
    print("Target:", target_example)

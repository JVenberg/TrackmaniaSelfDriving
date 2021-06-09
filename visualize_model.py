from inspect import trace
import torch
import hiddenlayer as hl

from model import TrackmaniaNet

trackmania_net = TrackmaniaNet()

visualize_graph = hl.build_graph(trackmania_net, torch.zeros([64, 1, 64, 64]), transforms=[
    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    hl.transforms.Fold("Linear > Relu", "LinearRelu")
])
visualize_graph.theme = hl.graph.THEMES["blue"].copy()

dot = visualize_graph.build_dot()
dot.format = 'png'
dot.attr("graph", rankdir='TB')
dot.render('model_diagram', directory='docs/imgs', cleanup=True)

# visualize_graph.save('docs/imgs/model_diagram.png', 'png')
function [gradients loss] = InputGradients_two_vectors_new_predict(dlnet,dlX,Y_ref,NameOutput)

Y_latent = predict(dlnet,dlX,'Output',NameOutput);

loss = sum((Y_latent-Y_ref).*(Y_latent-Y_ref));

gradients = dlgradient(loss,dlX);

end
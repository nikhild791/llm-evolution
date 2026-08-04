def train_simple_model(epoch,model,dataloader,loss,optimizer):
    for _ in range(epoch):
        model.train()
        for x,y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            lossx = loss(output, y)
            lossx.backward()
            optimizer.step()
        print(lossx)
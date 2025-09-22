from Dataset_processing import *

vision_train_dataset = MyDataset(root='D:\\cowhidedataset2\\train_data\\vision')
touch_train_dataset = MyDataset(root='D:\\cowhidedataset2\\train_data\\touch')
vision_test_dataset = MyDataset(root='D:\\cowhidedataset2\\test_data\\vision')
touch_test_dataset = MyDataset(root='D:\\cowhidedataset2\\test_data\\touch')
combined_train_dataset = CombinedDataset(vision_train_dataset, touch_train_dataset, train=True)
combined_test_dataset = CombinedDataset(vision_test_dataset, touch_test_dataset, train=False)
EMA_train_dataloader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
EMA_test_dataloader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False)

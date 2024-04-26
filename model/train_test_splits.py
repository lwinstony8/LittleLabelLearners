'''
# # Complete data
# full and full
train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
    my_dataloader.x_train, 
    my_dataloader.y_train, 
    my_dataloader.x_test, 
    my_dataloader.y_test)
'''


# # Train-subset data
# # 7 and full
# train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
#     my_dataloader.x_train_subset, 
#     my_dataloader.y_train_subset, 
#     my_dataloader.x_test, 
#     my_dataloader.y_test)


# # Train/Testsubset data
# # 7 and 7
# train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
#     my_dataloader.x_train_subset, 
#     my_dataloader.y_train_subset, 
#     my_dataloader.x_test_subset, 
#     my_dataloader.y_test_subset)

# # Train-subset data
# # full and 7
# train_dataset, labeled_train_dataset, test_dataset = my_dataloader.prepare_dataset(
#     my_dataloader.x_train, 
#     my_dataloader.y_train, 
#     my_dataloader.x_test_subset, 
#     my_dataloader.y_test_subset)


# print(f'{labeled_train_dataset=}')
# print(f'{test_dataset=}')
# exit()
import os
import sys
import shutil
import random
import csv

target_path = r'./WebApp/data/nii_base/124_lung_train/'
train_images = r"./WebApp/data/nii_base/124_lung/"

folder_list = os.listdir(train_images)
#rint(folder_list)
#
k = 30  # number of test images

index = [i for i in range(len(folder_list))]
random.shuffle(index)

#evalF = open(os.path.join(train_images+'BoneMarrow_folderlist.csv'), 'w')
#for i in range(len(folder_list)):
#    evalF.write('{},{}\n'.format(folder_list[index[i]], index[i]))
#    evalF.flush()

# filename = train_images+'BoneMarrow_folderlist.csv'
#
# with open(filename) as f:
#     reader = csv.reader(f)
#     header_row = next(reader)
#
#     highs = [header_row[1]]
#     for row in reader:
#         highs.append(row[1])
#
# print(highs)
#folder_list = ['2013E_19034', '2013E_19111', '2013E_19209', '2013E_19258', '2013E_19424', '2013E_19798', '2013E_19803', '2014E_29084', '2014E_29259', '2014E_29366', '2014E_29385', '2014E_29441', '2014E_29880', '2015E_39025', '2015E_39029', '2015E_39084', '2015E_39129', '2015E_39166', '2015E_39247', '2015E_39461', '2015E_39540', '2015E_39557', '2015E_40250', '2015E_40255', '2015E_40328', '2016E_48959', '2016E_49015', '2016E_49092', '2016E_49149', '2016E_50099', '2016E_50100', '2016E_50388', '2016E_50524', '2016M_49279', '2016M_49340', '2016M_49415', '2016M_49439', '2016M_49440', '2016M_49452', '2016M_49615', '2016M_49653', '2016M_49725', '2016M_49731', '2016M_49822', '2016M_49936', '2016M_50158', '2016M_50169', '2016M_50474', '2016M_50555', '2016M_50605', '2016M_50610', '2016M_50627', '2016M_50631', '2017E_59032', '2017E_59049', '2017E_59095', '2017E_59285', '2017E_59401', '2017E_59459', '2017E_59850', '2017E_60001', '2017E_60061', '2017E_60351', '2017E_60374', '2017E_60441', '2017E_60447', '2017E_60503', '2017E_60685', '2017E_61005', '2017M_58915', '2017M_58988', '2017M_59005', '2017M_59012', '2017M_59065', '2017M_59149', '2017M_59243', '2017M_59351', '2017M_59511', '2017M_59538', '2017M_59649', '2017M_59654', '2017M_59848', '2017M_59858', '2017M_59964', '2017M_59985', '2017M_60033', '2017M_60086', '2017M_60778RGGGT', '2017M_61004', '2018E_68996', '2018E_69002', '2018E_69009', '2018E_69035', '2018E_69344', '2018E_69445', '2018E_69665', '2018E_69813', '2018M_70574', '2018M_70717', 'R012745', 'R012746', 'R012814', 'R012833', 'R012848', 'R012895', 'R012912', 'R012934', 'R012999', 'R013003', 'RecentE_71098', 'RecentE_71279', 'RecentE_79261', 'RecentE_79694', 'RecentE_79743', 'RecentE_79776', 'RecentM_70860', 'RecentM_70940', 'RecentM_71007', 'RecentM_71056', 'RecentM_71191', 'RecentM_71327', 'RecentM_78890', 'RecentM_78926', 'RecentM_78968', 'RecentM_79007', 'RecentM_79113', 'RecentM_79378', 'RecentM_79432', 'RecentM_79669', 'RecentM_latest_79464', 'RecentM_latest_79469', 'RecentM_latest_79489', 'RecentM_latest_79526', 'RecentM_latest_79529', 'RecentM_latest_79558', 'RecentM_latest_79607', 'RecentM_latest_79692', 'RecentM_latest_79773', 'RecentM_latest_79791', 'RecentM_latest_79813', 'RecentM_latest_79827', 'RecentM_latest_79829', 'RecentM_latest_79886', 'RecentM_latest_79921', 'RecentM_latest_79930', 'RecentM_latest_79966', 'RecentM_latest_80054', 'Standard50M_60508', 'Standard50M_60808', 'Standard50M_60814']
#highs = ['40', '90', '34', '41', '86', '146', '70', '25', '135', '147', '5', '108', '89', '47', '114', '88', '96', '19', '62', '73', '24', '148', '125', '58', '104', '48', '138', '49', '103', '120', '133', '95', '118', '71', '9', '23', '140', '15', '43', '136', '64', '113', '112', '115', '123', '33', '3', '83', '1', '137', '127', '141', '121', '77', '44', '7', '76', '74', '36', '101', '131', '30', '51', '134', '100', '2', '117', '28', '11', '126', '37', '66', '82', '80', '119', '27', '122', '129', '53', '132', '0', '65', '29', '57', '106', '56', '91', '109', '107', '149', '20', '46', '111', '128', '67', '22', '72', '139', '78', '39', '61', '26', '84', '18', '68', '99', '21', '94', '92', '52', '55', '12', '93', '59', '17', '145', '54', '60', '116', '110', '142', '6', '79', '63', '31', '75', '81', '144', '130', '143', '8', '50', '124', '4', '13', '35', '97', '42', '105', '45', '16', '14', '32', '38', '102', '69', '98', '10', '87', '85']

for ii in range(k):
     shutil.copytree(train_images + folder_list[ii],
                     target_path + 'test/' + folder_list[ii])

#for ii in range(k):
#     shutil.copytree(train_images + folder_list[ii+k],
#                     target_path + 'val/' + folder_list[ii+k])

for ii in range(len(folder_list) - k):
     shutil.copytree(train_images + folder_list[ii+2*k],
                     target_path + 'train/' + folder_list[ii+k])






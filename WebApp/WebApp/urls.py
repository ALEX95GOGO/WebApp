from django.conf.urls import url, include
from django.views.static import serve
from django.conf.urls.static import static
from . import views, settings
from django.contrib import admin
import re
import os
from adduser import usersController

admin.autodiscover()

 
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', views.overview),
    url(r'^input_id/$', views.input_id),
    url(r'^check_existing_id/$', views.check_existing_id),
    url(r'^check_label/$', views.check_label),
    url(r'^choose_label/$', views.choose_label),
    url(r'^prepare_data/$', views.prepare_data),
    url(r'^plan/$', views.plan),
    url(r'^train/$', views.train),
    url(r'^predict/$', views.predict),
    url(r'^evaluate/$', views.evaluate),
    url(r'^check_result/', views.check_result),
    url(r'^mode/2d/', views.mode),
    url(r'^mode/3d_fullres/', views.mode),
    url(r'^check_result/', views.check_result),
    url(r'^upload/', views.upload),
    url(r'^upload_page/', views.upload_files),
    url(r'^download/', views.download_file),
    url(r'^toAddUserPage/', usersController.to_add_user_page),
    url(r'^addUser/', usersController.add_user),
    url(r'^toUpdateUserPage/', usersController.to_update_user_page),
    url(r'^updateUser/', usersController.update_user),
    url(r'^delUser/', usersController.del_user_by_id),
    url(r'^users/', usersController.user_list),
]

with open('./log/mode.file', "r") as f:
    data = f.read()
    s0 = re.split('/', data)
with open('./log/taskid.file', "r") as f:
    data = f.read()
    s1 = re.split('\n', data)
with open('./log/train_label.file', "r") as f:
    data = f.read()
    s2 = re.split('\n', data)


urlpatterns += static('/img/', document_root='{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/'.format(os.getenv('RESULTS_FOLDER'),s0[2],s1[0],s2[0]))

#urlpatterns += static('/img/', document_root='/home/data/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task155_LungL/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/')
urlpatterns += static('/prediction/', document_root='{}/compare/Task{}_{}/'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0]))



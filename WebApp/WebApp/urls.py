from django.conf.urls import url, include
from django.views.static import serve
from django.conf.urls.static import static
from . import views, settings
from django.contrib import admin
import re
admin.autodiscover()

 
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', views.overview),
    url(r'^import/$', views.import_data),
    url(r'^input_id/$', views.input_id),
    url(r'^check_existing_id/$', views.check_existing_id),
    url(r'^check_label/$', views.check_label),
    url(r'^choose_label/$', views.choose_label),
    url(r'^prepare_data/$', views.prepare_data),
    url(r'^plan/$', views.plan),
    url(r'^train/$', views.train),
    url(r'^predict/$', views.predict),
    url(r'^evaluate/$', views.evaluate),
    url(r'^progress/(?P<path>.*)$', serve, {'document_root': '/home/zhuoli.zhuang/nnUNet_data/nnUNet_trained_models/nnUNet/3d_fullres/Task125_DoubleLung/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/progress.png'}),
    url(r'^disk/', views.register),
    url(r'^check_result/', views.check_result),
    url(r'^mode/2d/', views.mode),
    url(r'^mode/3d_fullres/', views.mode),
    #url(r'^mode/3d_cascade/', views.mode),
    url(r'^check_result/', views.check_result),
    url(r'upload_page/', views.upload_files),
    url(r'upload/', views.upload),
    #url(r'^show_progress1/', views.show_progress1),
    #url(r'^show_progress/', views.show_progress),
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


urlpatterns += static('/img/', document_root='/home/zhuoli.zhuang/nnUNet_data/nnUNet_trained_models/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/'.format(s0[2],s1[0],s2[0]))
urlpatterns += static('/prediction/', document_root='/home/zhuoli.zhuang/nnUNet_data/compare/Task{}_{}/'.format(s1[0],s2[0]))

from django.conf.urls import url, include
from django.views.static import serve
from django.conf.urls.static import static
from . import views, settings
from django.contrib import admin
import re
admin.autodiscover()

 
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', views.import_data),
    url(r'^input_id/$', views.input_id),
    url(r'^check_label/$', views.check_label),
    url(r'^choose_label/$', views.choose_label),
    url(r'^prepare_data/$', views.prepare_data),
    url(r'^plan/$', views.plan),
    url(r'^train/$', views.train),
    url(r'^predict/$', views.predict),
    url(r'^evaluate/$', views.evaluate),
    #url(r'^progress/(?P<path>.*)$', serve, {'document_root': '/home/img/data/nnUNet_trained_models/nnUNet/3d_fullres/Task089_Pelvis_lp/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/progress.png'}),
    url(r'^disk/', views.register),
    url(r'^check_result/', views.check_result),
    url(r'^show_progress1/', views.show_progress1),
    url(r'^show_progress/', views.show_progress),
]

with open('./log/taskid.file', "r") as f:
    data = f.read()
    s1 = re.split('\n', data)
with open('./log/train_label.file', "r") as f:
    data = f.read()
    s2 = re.split('\n', data)
urlpatterns += static('/img/', document_root='/home/img/data/nnUNet_trained_models/nnUNet/2d/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/'.format(s1[0],s2[0]))
urlpatterns += static('/prediction/', document_root='/home/img/data/compare/Task{}_{}/'.format(s1[0],s2[0]))

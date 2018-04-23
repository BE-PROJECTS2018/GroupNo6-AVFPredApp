from django.conf.urls import url
from django.urls import path
from django.contrib.auth import views as auth_views
from .views import UserHome,custom_login
from . import views

app_name='dialysisgui'
urlpatterns =[
	
	url(r'^$', views.redirect_home ,name='redirect'),
	url(r'^input/$',UserHome.as_view(), name='input'),
	url(r'^upload/$', views.model_form_upload ,name='upload'),
	url(r'^train/$', views.model_form_train ,name='train'),
	url(r'^patient_his/$',views.patient_history ,name='patient_history'),
	url(r'^patient/$',views.patient_details ,name='patient'),
	#url(r'^login_success/$', views.login_success ,name='login_success'),
	url(r'^admin_home/', auth_views.login, {'template_name': 'admin-home.html'}, name='admin_home'),
	url(r'^home/', auth_views.login, {'template_name': 'home.html'}, name='home'),
	url(r'^graph/', auth_views.login, {'template_name': 'graph.html'}, name='ga'),
	url(r'^login/$', custom_login , {'template_name': 'login.html'}, name='login'),
	url(r'^about/$', auth_views.login, name='about'),
	url(r'^bar/$', views.get_bar, name='bar'),
	url(r'^bar1/$', views.get_ga, name='bar1'),
	url(r'^redirect/$', views.redirect_home, name='redirect'),
]
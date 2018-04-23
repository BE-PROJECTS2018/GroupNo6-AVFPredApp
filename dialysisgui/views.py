# importing required modules
import PyPDF2
import os
import csv
import urllib.request
from PyPDF2 import merger, PdfFileReader 
from django.shortcuts import render,redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import TemplateView
from django.contrib.auth.views import login
from django.conf import settings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import io
import csv
import urllib
import urllib.request
import ast
from django.contrib import messages
from .userform import DataForm,DocumentForm
from .p2 import mlp
from .p1 import mlp_csv
from .algo import random_forest_csv,svm_csv,mlp_csv
from django.views.generic import View
from .models import parameter
from django.template.loader import get_template
from .utils import render_to_pdf #created in step 4
import datetime, time
BASE_DIR=getattr(settings,"BASE_DIR",None)
#ax1=plt.subplot(1,1,1)

def redirect_home(request):
	if request.user.is_authenticated:
		if request.user.is_staff:
			return redirect('dialysisgui:admin_home')
		else:
			return redirect('dialysisgui:home')
	else:
		return redirect('dialysisgui:login')

def custom_login(request, **kwargs):
	if request.user.is_authenticated:
		return redirect(settings.LOGIN_REDIRECT_URL)
	else:
		return login(request,**kwargs)

def get_bar(request):
	left = [1, 2, 3]

	f=open('finalized_model_mlp.sav', 'rb')
	cl_mlp=pickle.load(f)
	accuracy_mlp=pickle.load(f)

	f1=open('finalized_model_svm.sav','rb')
	cl_svm=pickle.load(f1)
	accuracy_svm=pickle.load(f1)

	f2=open('finalized_model_rfc.sav','rb')
	cl_rfc=pickle.load(f2)
	accuracy_rfc=pickle.load(f2)

	height = [accuracy_svm*100, accuracy_rfc*100, accuracy_mlp*100]
	tick_label = ['SVM', 'RFC', 'MLP']
	plt.subplot('232')
	plt.bar(left, height, tick_label = tick_label, width = 0.4, color = ['blue','red','green'])
	plt.xlabel('Algorithms')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Test')
	f1 = io.BytesIO()
	plt.savefig(f1, format='png')
	return HttpResponse(f1.getvalue(), content_type='image/png')

def get_ga(request):
	f1 = io.BytesIO()
	# f1=open('finalized_model_rfc.sav', 'rb')
	# f2=open('finalized_model_svm.sav', 'rb')
	df =  pd.read_csv('traintest.csv')
	temp=np.array(df.Class)
	temp2=np.count_nonzero(temp == 1)
	temp3=temp.size-temp2

	labels = 'Training data', 'Testing data'
	sizes = [80,20]
	colors = ['blue', 'red']
	explode = (0, 0)
	ax=plt.subplot(431)
	plt.rcParams['font.size'] = 7.0
	ax.set_title('Total Data')
	ttl=ax.title
	ttl.set_position([.5,1])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	labels = 'Bad Fistulae', 'Good Fistulae'
	sizes = [temp2,temp3]
	plt.rcParams['font.size'] = 7.0
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(432)
	colors = ['red', 'blue']
	ax.set_title('Training Data')
	ttl=ax.title
	ttl.set_position([.5,1])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=40)

	labels = 'Bad Fistulae', 'Good Fistulae'
	sizes = [temp2,temp3]
	plt.rcParams['font.size'] = 7.0
	colors = ['red', 'blue']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(433)
	ax.set_title('Testing Data')
	ttl=ax.title
	ttl.set_position([.5,1])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=40)

	f=open('finalized_model_mlp.sav', 'rb')
	cl_mlp=pickle.load(f)
	accuracy_mlp=pickle.load(f)
	a=pickle.load(f)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[0][0],a[0][1]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(434)
	ax.set_title('MLP:          Good Fistulae')
	ttl=ax.title
	ttl.set_position([.2,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[1][1],a[1][0]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(435)
	ax.set_title('Bad Fistulae')
	ttl=ax.title
	ttl.set_position([.4,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=180)

	labels = 'Accuracy', ''
	plt.rcParams['font.size'] = 7.0
	sizes = [accuracy_mlp,1-accuracy_mlp]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(436)
	plt.axis('equal')
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	d=open('finalized_model_rfc.sav', 'rb')
	cl_mlp=pickle.load(d)
	accuracy_mlp=pickle.load(d)
	a=pickle.load(d)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[0][0],a[0][1]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(437)
	ax.set_title('RFC:          Good Fistulae')
	ttl=ax.title
	ttl.set_position([.2,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[1][1],a[1][0]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(438)
	ax.set_title('Bad Fistulae')
	ttl=ax.title
	ttl.set_position([.4,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=180)

	labels = 'Accuracy', ''
	plt.rcParams['font.size'] = 7.0
	sizes = [accuracy_mlp,1-accuracy_mlp]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(439)
	plt.axis('equal')
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	r=open('finalized_model_svm.sav', 'rb')
	cl_mlp=pickle.load(r)
	accuracy_mlp=pickle.load(r)
	a=pickle.load(r)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[0][0],a[0][1]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(4,3,10)
	ax.set_title('SVM:          Good Fistulae')
	ttl=ax.title
	ttl.set_position([.2,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	labels = 'Right Prediction', 'Wrong Prediction'
	plt.rcParams['font.size'] = 7.0
	sizes = [a[1][1],a[1][0]]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(4,3,11)
	ax.set_title('Bad Fistulae')
	ttl=ax.title
	ttl.set_position([.4,0.9])
	plt.axis('equal')
	plt.rcParams['font.size'] = 5.0
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=180)

	labels = 'Accuracy', ''
	plt.rcParams['font.size'] = 7.0
	sizes = [accuracy_mlp,1-accuracy_mlp]
	colors = ['blue', 'red']
	explode = (0, 0)  # explode 1st slice
	ax=plt.subplot(4,3,12)
	plt.axis('equal')
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	        autopct='%1.0f%%', shadow=False, startangle=150)

	plt.savefig(f1, format='png')
	plt.clf()
    	#plt.show()
	return HttpResponse(f1.getvalue(), content_type='image/png')



class UserHome(TemplateView):
	template_name = 'input.html'

	def get(self, request):
		form1=DataForm()
		return render(request, self.template_name, {'form1' : form1})

	def post(self, request):
		form1 = DataForm(request.POST)
		if form1.is_valid():
			form1.save()
			sid=form1.cleaned_data['sid']
			pid=form1.cleaned_data['pid']
			hrmax=form1.cleaned_data['hrmax']
			hrmin=form1.cleaned_data['hrmin']
			hrmaxmin=form1.cleaned_data['hrmaxmin']
			hrmean=form1.cleaned_data['hrmean']
			hrstd=form1.cleaned_data['hrstd']
			sbpmax=form1.cleaned_data['sbpmax']
			sbpmin=form1.cleaned_data['sbpmin']
			sbpmaxmin=form1.cleaned_data['sbpmaxmin']
			sbpmean=form1.cleaned_data['sbpmean']
			sbpstd=form1.cleaned_data['sbpstd']
			dbpmax=form1.cleaned_data['dbpmax']
			dbpmin=form1.cleaned_data['dbpmin']
			dbpmaxmin=form1.cleaned_data['dbpmaxmin']
			dbpmean=form1.cleaned_data['dbpmean']
			dbpstd=form1.cleaned_data['dbpstd']
			vpmax=form1.cleaned_data['vpmax']
			vpmin=form1.cleaned_data['vpmin']
			vpmaxmin=form1.cleaned_data['vpmaxmin']
			vpmean=form1.cleaned_data['vpmean']
			vpstd=form1.cleaned_data['vpstd']
			apmax=form1.cleaned_data['apmax']
			apmin=form1.cleaned_data['apmin']
			apmaxmin=form1.cleaned_data['apmaxmin']
			apmean=form1.cleaned_data['apmean']
			apstd=form1.cleaned_data['apstd']
			ktvslope=form1.cleaned_data['ktvslope']
			tbvslope=form1.cleaned_data['tbvslope']
			ts = time.time()
			time1=datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
			user=request.user.username
			
			pred = mlp(hrmax,hrmin,hrmaxmin,hrmean,hrstd,sbpmax,sbpmin,sbpmaxmin,sbpmean,sbpstd,dbpmax,dbpmin,dbpmaxmin,dbpmean,dbpstd,vpmax,vpmin,vpmaxmin,vpmean,vpstd,apmax,apmin,apmaxmin,apmean,apstd,ktvslope,tbvslope)
			args = {'form1' : form1, 'pred' : pred}
			form1=DataForm()
			#response = HttpResponse(content_type='text/csv')
			#response['Content-Disposition'] ='attachment; filename = "test.csv"'
			f= open('test.csv','a',newline='')
			writer = csv.writer(f, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
			#writer.writerow(['sid','pid','hrmax','hrmin','hrmaxmin','hrmean','hrstd','sbpmax','sbpmin','sbpmaxmin','sbpmean','sbpstd','dbpmax','dbpmin','dbpmaxmin','dbpmean','dbpstd','vpmax','vpmin','vpmaxmin','vpmean','vpstd','apmax','apmin','apmaxmin','apmean','apstd','ktvslope','tbvslope','class'])
			writer.writerow([sid,pid,hrmax,hrmin,hrmaxmin,hrmean,hrstd,sbpmax,sbpmin,sbpmaxmin,sbpmean,sbpstd,dbpmax,dbpmin,dbpmaxmin,dbpmean,dbpstd,vpmax,vpmin,vpmaxmin,vpmean,vpstd,apmax,apmin,apmaxmin,apmean,apstd,ktvslope,tbvslope,pred,time1,user])
			#para = parameter.objects.all().values_list('sid','pid','hrmax','hrmin','hrmaxmin','hrmean','hrstd','sbpmax','sbpmin','sbpmaxmin','sbpmean','sbpstd','dbpmax','dbpmin','dbpmaxmin','dbpmean','dbpstd','vpmax','vpmin','vpmaxmin','vpmean','vpstd','apmax','apmin','apmaxmin','apmean','apstd','ktvslope','tbvslope')
			template = get_template('report.html')
			context = {
            'sid': sid,
            'pid':  pid,
            'hrmax': hrmax,
            'hrmin': hrmin,
            'hrmaxmin': hrmaxmin,
            'hrmean': hrmean,
            'hrstd': hrstd,
            'sbpmax': sbpmax,
            'sbpmin': sbpmin,
            'sbpmaxmin': sbpmaxmin,
            'sbpmean': sbpmean,
            'sbpstd': sbpstd,
            'dbpmax': dbpmax,
            'dbpmin': dbpmin,
            'dbpmaxmin': dbpmaxmin,
            'dbpmean': dbpmean,
            'dbpstd': dbpstd,
            'vpmax': vpmax,
            'vpmin': vpmin,
            'vpmavmin': vpmaxmin,
            'vpmean': vpmean,
            'vpstd': vpstd,
            'apmax': apmax,
            'apmin': apmin,
            'apmaxmin': apmaxmin,
            'apmean' : apmean,
            'apstd' : apstd,
            'ktvslope' : ktvslope,
            'tbvslope' : tbvslope,
			'pred' : pred,
        }
		html = template.render(context)
		pdf = render_to_pdf('report.html', context)
		if pdf:
			response = HttpResponse(pdf, content_type='application/pdf')
			filename = "report_%s.pdf" %(pid)
			content = "inline; filename='%s'" %(filename)
			download = request.GET.get("download")
			if download:
				content = "attachment; filename='%s'" %(filename)
			response['Content-Disposition'] = content
			return response
		return HttpResponse("Not found")
		#return render(request,self.template_name,{'form1' : form1})		

def model_form_upload(request):
	if request.method == 'POST':
		form2=DocumentForm(request.POST, request.FILES)
		if form2.is_valid():
			form2.save()
			csv_file=request.FILES['document']
			if not csv_file.name.endswith('.csv'):
				messages.error(request,'File is not csv type')
				return render(request, 'upload.html',{'form2' : form2})
			file=csv_file.name
			

			

			input_file = open(file,"r+")
			reader_file = csv.reader(input_file)
			value = len(list(reader_file))
			value=value-1

			df_file = pd.read_csv(file, header = 0)
			X_file = np.array(df_file)
			merger=PyPDF2.PdfFileMerger()
			for i in range(0, value):
				sid1=X_file[i][0]
				pid1=X_file[i][1]
				hrmax1=X_file[i][2]
				hrmin1=X_file[i][3]
				hrmaxmin1=X_file[i][4]
				hrmean1=X_file[i][5]
				hrstd1=X_file[i][6]
				sbpmax1=X_file[i][7]
				sbpmin1=X_file[i][8]
				sbpmaxmin1=X_file[i][9]
				sbpmean1=X_file[i][10]
				sbpstd1=X_file[i][11]
				dbpmax1=X_file[i][12]
				dbpmin1=X_file[i][13]
				dbpmaxmin1=X_file[i][14]
				dbpmean1=X_file[i][15]
				dbpstd1=X_file[i][16]
				vpmax1=X_file[i][17]
				vpmin1=X_file[i][18]
				vpmaxmin1=X_file[i][19]
				vpmean1=X_file[i][20]
				vpstd1=X_file[i][21]
				apmax1=X_file[i][22]
				apmin1=X_file[i][23]
				apmaxmin1=X_file[i][24]
				apmean1=X_file[i][25]
				apstd1=X_file[i][26]
				ktvslope1=X_file[i][27]
				tbvslope1=X_file[i][28]
				pred1=mlp(hrmax1,hrmin1,hrmaxmin1,hrmean1,hrstd1,sbpmax1,sbpmin1,sbpmaxmin1,sbpmean1,sbpstd1,dbpmax1,dbpmin1,dbpmaxmin1,dbpmean1,dbpstd1,vpmax1,vpmin1,vpmaxmin1,vpmean1,vpstd1,apmax1,apmin1,apmaxmin1,apmean1,apstd1,ktvslope1,tbvslope1)
				args = {'form2' : form2, 'pred' : pred1}
				ts = time.time()
				time1=datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
				user=request.user.username
				form2=DataForm()
				f= open('test.csv','a',newline='')
				writer = csv.writer(f, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
				#writer.writerow(['sid','pid','hrmax','hrmin','hrmaxmin','hrmean','hrstd','sbpmax','sbpmin','sbpmaxmin','sbpmean','sbpstd','dbpmax','dbpmin','dbpmaxmin','dbpmean','dbpstd','vpmax','vpmin','vpmaxmin','vpmean','vpstd','apmax','apmin','apmaxmin','apmean','apstd','ktvslope','tbvslope','class'])
				writer.writerow([sid1,pid1,hrmax1,hrmin1,hrmaxmin1,hrmean1,hrstd1,sbpmax1,sbpmin1,sbpmaxmin1,sbpmean1,sbpstd1,dbpmax1,dbpmin1,dbpmaxmin1,dbpmean1,dbpstd1,vpmax1,vpmin1,vpmaxmin1,vpmean1,vpstd1,apmax1,apmin1,apmaxmin1,apmean1,apstd1,ktvslope1,tbvslope1,pred1,time1,user])
				#para = parameter.objects.all().values_list('sid','pid','hrmax','hrmin','hrmaxmin','hrmean','hrstd','sbpmax','sbpmin','sbpmaxmin','sbpmean','sbpstd','dbpmax','dbpmin','dbpmaxmin','dbpmean','dbpstd','vpmax','vpmin','vpmaxmin','vpmean','vpstd','apmax','apmin','apmaxmin','apmean','apstd','ktvslope','tbvslope')
				template = get_template('report.html')
				dic2 = {
            	'sid': sid1,
            	'pid':  pid1,
            	'hrmax': hrmax1,
            	'hrmin': hrmin1,
            	'hrmaxmin': hrmaxmin1,
            	'hrmean': hrmean1,
            	'hrstd': hrstd1,
            	'sbpmax': sbpmax1,
            	'sbpmin': sbpmin1,
            	'sbpmaxmin': sbpmaxmin1,
            	'sbpmean': sbpmean1,
            	'sbpstd': sbpstd1,
            	'dbpmax': dbpmax1,
            	'dbpmin': dbpmin1,
            	'dbpmaxmin': dbpmaxmin1,
            	'dbpmean': dbpmean1,
            	'dbpstd': dbpstd1,
            	'vpmax': vpmax1,
            	'vpmin': vpmin1,
            	'vpmavmin': vpmaxmin1,
            	'vpmean': vpmean1,
            	'vpstd': vpstd1,
            	'apmax': apmax1,
            	'apmin': apmin1,
            	'apmaxmin': apmaxmin1,
            	'apmean' : apmean1,
            	'apstd' : apstd1,
            	'ktvslope' : ktvslope1,
            	'tbvslope' : tbvslope1,
				'pred' : pred1,
        		}
				html = template.render(dic2)
				pdf = render_to_pdf('report.html', dic2)
				response = HttpResponse(pdf, content_type='application/pdf')
				#with open('Report{0}.pdf'.format(i+1), 'wb') as f:
				#	f.write(response.content)
				#merger.append(PdfFileReader('Report{0}.pdf'.format(i+1), 'rb'))	
			#merger.write('Results.pdf')
			file_path = os.path.join(BASE_DIR,"Results.pdf") #change this according to your machine
			if os.path.exists(file_path):
				with open(file_path, 'rb') as fh:
					response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
					response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
					return response
	else:
		form2=DocumentForm()
	return render(request, 'upload.html',{'form2' : form2})			

#for training:
def model_form_train(request):
	if request.method == 'POST':
		form2=DocumentForm(request.POST, request.FILES)
		if form2.is_valid():
			form2.save()
			csv_file=request.FILES['document']
			if not csv_file.name.endswith('.csv'):
				messages.error(request,'File is not csv type')
				return render(request, "training.html" ,{'form2' : form2})
			file=csv_file.name
			
			#x = file[:-4]
			#ts = time.time()
			#x = x + datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')+".csv"
			#os.rename(file,x)

			input_file = open(file,"r+")
			reader_file = csv.reader(input_file)
			value = len(list(reader_file))
			value=value-1

			df_file = pd.read_csv(file, header = 0)
			X_file = np.array(df_file)
			a = request.POST['drop1']
			if a=="1":
				accuracy = random_forest_csv(file)
				accuracy = accuracy*100
			elif a=="2":
				accuracy = svm_csv(file)
				accuracy = accuracy*100
			elif a=="3":
				accuracy = mlp_csv(file)
				accuracy = accuracy*100
			else:
				accuracy = "error"
		return render(request, "training.html" ,{'form2' : form2, 'accuracy' : accuracy})
	else:
		form2=DocumentForm(request.POST, request.FILES)
		return render(request, "training.html" ,{'form2' : form2})

def patient_details(request):
	csv_preform = []
	if request.method=='POST':
		a=request.POST['pid']
		# reader = csv.DictReader(f)
		# Get a specific object
		#data_source = Source.objects.get(id=source_id)
		#context_dict['reader'] = reader

		# Open the csv and print it to terminal
		with open("test.csv",newline='\n') as f:
			include = [0, 1, 29, 30, 31]
			
			spamreader = csv.reader(f, delimiter=',', quotechar='|')
			for row in spamreader:
				content = list(row[i] for i in include)
				if content[1] == a:
					csv_preform.append(content)
					print(content)
			print(csv_preform)
	return render(request, 'patients data.html', {'csv_preform' : csv_preform})
	
def patient_history(request):
	csv_preform = []
	a=request.user.username
		# reader = csv.DictReader(f)
		# Get a specific object
		#data_source = Source.objects.get(id=source_id)
		#context_dict['reader'] = reader

		# Open the csv and print it to terminal
	with open("test.csv",newline='\n') as f:
		include = [0, 1, 29, 30, 31]
			
		spamreader = csv.reader(f, delimiter=',', quotechar='|')
		for row in spamreader:
			content = list(row[i] for i in include)
			if content[4] == a:
				csv_preform.append(content)
				print(content)
		print(csv_preform)
	return render(request, 'home.html', {'csv_preform' : csv_preform})
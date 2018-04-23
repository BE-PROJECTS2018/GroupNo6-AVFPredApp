from django import forms
from .models import parameter,Document


class DataForm(forms.ModelForm):
	sid=forms.CharField(max_length=100, label='Session ID')
	pid=forms.CharField(max_length=8, label='Patient ID')
	hrmax=forms.FloatField(label='Max Heart Rate')
	hrmin=forms.FloatField(label='Min Heart Rate')
	hrmaxmin=forms.FloatField(label='Max-Min Heart Rate')
	hrmean=forms.FloatField(label='Mean Heart Rate')
	hrstd=forms.FloatField(label='Std Heart Rate')
	sbpmax=forms.FloatField(label='Max SBP')
	sbpmin=forms.FloatField(label='Min SBP')
	sbpmaxmin=forms.FloatField(label='Max-Min SBP')
	sbpmean=forms.FloatField(label='Mean SBP')
	sbpstd=forms.FloatField(label='Std SBP')
	dbpmax=forms.FloatField(label='Max DBP')
	dbpmin=forms.FloatField(label='Min DBP')
	dbpmaxmin=forms.FloatField(label='Max-Min DBP')
	dbpmean=forms.FloatField(label='Mean DBP')
	dbpstd=forms.FloatField(label='Std DBP')
	vpmax=forms.FloatField(label='Max VP')
	vpmin=forms.FloatField(label='Min VP')
	vpmaxmin=forms.FloatField(label='Max-Min VP')
	vpmean=forms.FloatField(label='Mean VP')
	vpstd=forms.FloatField(label='Std VP')
	apmax=forms.FloatField(label='Max AP')
	apmin=forms.FloatField(label='Min AP')
	apmaxmin=forms.FloatField(label='Max-Min AP')
	apmean=forms.FloatField(label='Mean AP')
	apstd=forms.FloatField(label='Std AP')
	ktvslope=forms.FloatField(label='KTV Slope')
	tbvslope=forms.FloatField(label='TBV Slope')

	class Meta:
		model = parameter
		fields = ('sid','pid','hrmax','hrmin','hrmaxmin','hrmean','hrstd','sbpmax','sbpmin','sbpmaxmin','sbpmean','sbpstd','dbpmax','dbpmin','dbpmaxmin','dbpmean','dbpstd','vpmax','vpmin','vpmaxmin','vpmean','vpstd','apmax','apmin','apmaxmin','apmean','apstd','ktvslope','tbvslope')

class DocumentForm(forms.ModelForm):
	document=forms.FileField(label ='Choose file: ')
	class Meta:
		model=Document
		fields=('description','document',)


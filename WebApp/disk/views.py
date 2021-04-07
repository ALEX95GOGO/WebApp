from django.shortcuts import render,render_to_response

# Create your views here.
def register(request):
    return render_to_response('register.html',{})
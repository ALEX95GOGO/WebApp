# -*- coding: utf-8 -*-
 
from django.shortcuts import render
from django.shortcuts import redirect
import sqlite3
from .models import User 


def to_add_user_page(request):
    print("Current user page")
    return render(request, "adduser.html")


def add_user(request):
    request.encoding = 'utf-8'
    if 'username' in request.POST and request.POST['username']:
        username = request.POST["username"]
    if 'password' in request.POST and request.POST['password']:
        password = request.POST["password"]
    if username and password:
        print("Current data added: username:{0},password:{1}".format(username, password))
        try:
            conn = sqlite3.connect("db.sqlite3")
            cursor = conn.cursor()
            cursor.execute("insert into adduser_user(username,password) values(?,?)", (username, password))
            conn.commit()
        except Exception as e:
            print("Error: {0}".format(e))
        finally:
            close_database(cursor, conn)
        print("Add User Info")
    else:
        print("No data")
    return redirect(to="../users/")


def to_update_user_page(request):
    model = {}
    if "id" in request.GET and request.GET["id"]:
        id = request.GET["id"]
    if id:
        try:
            conn = sqlite3.connect("db.sqlite3")
            cursor = conn.cursor()
            cursor.execute("select * from adduser_user where id=?", id)
            user_data = cursor.fetchone()
            model["updateUser"] = convert_to_user(user_data)
        except Exception as e:
            print("Error:{0}".format(e))
        finally:
            close_database(cursor, conn)
    else:
        print("No ID")
    return render(request, "updateUser.html", model)


def update_user(request):
    if "id" in request.POST and request.POST['id']:
        id = request.POST["id"]
        username = request.POST["username"]
        password = request.POST["password"]
        try:
            conn = sqlite3.connect("db.sqlite3")
            cursor = conn.cursor()
            cursor.execute("update adduser_user set username=? ,password=? where id=?", (username, password,
                                                                                  id))
            conn.commit()
        except Exception as e:
            print("Error: {0}".format(e))
        finally:
            close_database(cursor, conn)
    else:
        print("No ID")
    return redirect(to="../users/")


def del_user_by_id(request):
    if "id" in request.GET and request.GET['id']:
        id = request.GET["id"]
        try:
            conn = sqlite3.connect("db.sqlite3")
            cursor = conn.cursor()
            cursor.execute("delete from adduser_user where id=?", id)
            conn.commit()
        except Exception as e:
            print("Error: {0}".format(e))
        finally:
            close_database(cursor, conn)
    else:
        print("No ID")
    return redirect(to="../users/")


def user_list(request):
    print("Current User Info")
    try:
        conn = sqlite3.connect("db.sqlite3")
        cursor = conn.cursor()
        cursor.execute("select * from adduser_user")
        users = [convert_to_user(item) for item in cursor.fetchall()]
        model = {"users": users}
        return render(request, "index.html", model)
    except Exception as e:
        print("Exception: {0}".format(e))
    finally:
        close_database(cursor, conn)


def convert_to_user(user_data=[]):
    return User(user_data[0], user_data[1], user_data[2])


def close_database(cursor, conn):
    if cursor in locals():
        cursor.close()
    if conn in locals():
        conn.close()

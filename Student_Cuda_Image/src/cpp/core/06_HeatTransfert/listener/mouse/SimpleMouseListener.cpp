#include <iostream>

#include "SimpleMouseListener.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*-----------------------*\
|*	Constructor  	*|
 \*----------------------*/

SimpleMouseListener::SimpleMouseListener()
    {
    //rien
    }

SimpleMouseListener::~SimpleMouseListener()
    {
    //rien
    }

/*-----------------------*\
|*	Methode  	*|
 \*----------------------*/

void SimpleMouseListener::onMouseMoved(const MouseEvent& event)
    {
    cout<<"[MouseMoved] : ";
    printXY(event);
    }

void SimpleMouseListener::onMousePressed(const MouseEvent& event)
    {
    cout<<"[MousePressed]  :  button="<<event.getButtonType()<< " : ";
    printXY(event);
    }

void SimpleMouseListener::onMouseReleased(const MouseEvent& event)
    {
    cout<<"[MouseReleased] :  button="<<event.getButtonType()<< " : ";
    printXY(event);
    }

void SimpleMouseListener::onMouseWheel(const MouseWheelEvent& event)
    {
    cout<<"[MouseWheel]    :  direction="<<event.getDirection()<< " : ";
    printXY(event);
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * (x,y) : coordonnée image, et non frame!
 */
void SimpleMouseListener::printXY(const MouseEvent& event)
    {
    cout<<" at pixel (x,y) = ("<<event.getX()<<","<<event.getY()<<")"<<endl;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/


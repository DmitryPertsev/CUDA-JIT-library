#ifndef __QUEUE_H__
#define __QUEUE_H__

template <class T>
class Queue
{
public:
	struct Data
	{
		T *data;
		Data *next;
		Data *prev;
	};

private:
	Data *first;
	Data *last;
	int elementCount;

public:
	Queue(void)
	{
		first = last = NULL;
		elementCount = 0;
	};
	~Queue(void)
	{
		while (first != last)
		{
			Data *cur = first;
			first = first->next;
			free(cur);
		}
		if (first != NULL)
			free(first);
		elementCount = 0;
	};
	void clearQueue(void)
	{
		while (first != last)
		{
			Data *cur = first;
			first = first->next;
			free(cur);
		}
		if (first != NULL)
			free(first);
		first = last = NULL;
		elementCount = 0;
	};
	void push(T *element)
	{
		Data *newElement = (Data *)malloc(sizeof(Data));
		newElement->next = NULL;
		newElement->prev = last;
		newElement->data = element;
		if (last == NULL)
		{
			last = newElement;
			first = newElement;
		}
		else
		{
			last->next = newElement;
			last = newElement;
		}
		//elementCount++;
	};
	T *pop(int position)
	{
		if (first == NULL)
			return NULL;
		Data *cur = first;
		// перемещаемся на позицию, которая была завершена
		for (int i = 0; i < position; i++)
			cur = cur->next;
		//first = first->next;
		//if (first == NULL)
		//{
		//	last = NULL;
		//	elementCount = 0;
		//}
		//else
		//	elementCount--;
		return cur->data;
	}
	int getCount()
	{
		return elementCount;
	}
};

#endif
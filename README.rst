|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Выявление поляризации текстов в новостном потоке
    :Тип научной работы: M1P
    :Автор: Роман Артемович Авдеев
    :Научный руководитель: д.ф.-м.н., Воронцов Константин Вячеславович
    :Научный консультант(при наличии): Ватолин Алексей

Abstract
========

В данной работе предлагается способ определения поляризации текстов в новостном потоке.
Решение основано на методах машинного обучения без учителя, что позволяет работать как с
малыми, так и с большими наборами текстов. Решается задача разделения множества новостных
сообщений на кластеры-мнения, выделения отдельных кластеров нейтральных и нерелевантных
документов. Предложены метрики оценивания качества отсева нерелевантных и нейтральных
сообщений. Реализована модель, работающая в среднем не хуже, чем разметчики.
Эксперименты проводились на датасете, состоящем из 30 корпусов новостных сообщений по
темам «политика» и «происшествия».

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.

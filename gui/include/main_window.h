#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QBoxPlotSeries>

#ifdef QT_CHARTS_USE_NAMESPACE
QT_CHARTS_USE_NAMESPACE
#endif

#include <set>
#include <map>
#include <memory>

#include "kernel_param.h"
#include "ptx_executor.h"

class QLineEdit;
class QGroupBox;
class CodeEditor;
class QToolBar;
class QTimer;
class QValueAxis;
class QCategoryAxis;

class SyntaxStyle;
class MainWindow;

class CUDAPTXPair : public QObject
{
  Q_OBJECT

public:
  CUDAPTXPair(const QString &name, MainWindow *main_window);

  void load_style(const QString &path, MainWindow *main_window);
  std::vector<KernelParameter> get_params();
  std::vector<Measurement> execute(PTXExecutor *executor);

  QTimer *timer {};
  QLineEdit *options {};
  CodeEditor *cuda {};
  CodeEditor *ptx {};

  MainWindow *main_window;

public slots:
  void reset_timer();

private slots:
  void regen_ptx();
};

class ScatterLineSeries
{
public:
    QLineSeries *line_series {};
    QScatterSeries *scatter_series {};

    ScatterLineSeries();

    void set_color(QColor color);
    void add_to_chart(QValueAxis *y_axis,
                      QCategoryAxis *x_axis,
                      QChart *chart);

    void append(int x, float y);
};

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();
  ~MainWindow();

  QToolBar *tool_bar {};

  std::vector<std::unique_ptr<CUDAPTXPair>> cuda_ptx_pairs;

  int execution_id {};

  ScatterLineSeries median_series;

  QChartView *chart_view {};
  QChart *chart {};
  QValueAxis *y_axis {};
  QCategoryAxis *x_axis {};

  float min_elapsed = std::numeric_limits<float>::max();
  float max_elapsed = 0.0f;

  SyntaxStyle* syntaxStyle;

  QAction *add_action {};
  QAction *run_action {};
  QAction *interpret_action {};

  std::unique_ptr<PTXExecutor> executor;

private slots:
  void add();
  void interpret();
  void execute();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H

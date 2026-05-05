import React, { useState } from 'react';
import { 
  Layout, Card, Row, Col, Statistic, Form, InputNumber, Button, Typography, Space, Divider, Select 
} from 'antd';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  DashboardOutlined, ExperimentOutlined, RocketOutlined, ArrowUpOutlined, ArrowDownOutlined, DatabaseOutlined, RobotOutlined 
} from '@ant-design/icons';
import { getPrediction } from './services/api';
import { PriceFeatures, PredictionResponse } from './types';
import './App.css';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

const App: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const initialValues: PriceFeatures = {
    price_lag_0: 23000,
    brent_lag_0: 85.5,
    brent_lag_7: 82.0,
    brent_mean_30: 83.2,
    regime: 0,
    brent_diff_1: 0.5,
    brent_diff_7: 3.5,
    days_since_last_change: 5,
    model_name: 'random_forest'
  };

  const onFinish = async (values: PriceFeatures) => {
    setLoading(true);
    try {
      const result = await getPrediction(values);
      setPrediction(result);
    } catch (error) {
      console.error("Prediction Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const chartData = [
    { name: 'Last Period', price: 22800 },
    { name: 'Current', price: initialValues.price_lag_0 },
    { name: 'Forecast', price: prediction ? prediction.predicted_price_tomorrow : initialValues.price_lag_0 },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', display: 'flex', alignItems: 'center' }}>
        <Title level={3} style={{ color: '#fff', margin: 0 }}>
          <RocketOutlined /> FuelPrice AI Enterprise
        </Title>
      </Header>
      
      <Layout>
        <Sider width={380} style={{ background: '#fff', padding: '24px' }}>
          <Title level={4}><ExperimentOutlined /> Control Panel</Title>
          <Form
            form={form}
            layout="vertical"
            initialValues={initialValues}
            onFinish={onFinish}
          >
            <Form.Item name="model_name" label={<strong>AI MODEL</strong>}>
              <Select size="large">
                <Option value="random_forest">Random Forest</Option>
                <Option value="xgboost">XGBoost</Option>
                <Option value="ensemble">Ensemble</Option>
              </Select>
            </Form.Item>
            <Row gutter={16}>
              {Object.keys(initialValues).filter(k => k !== 'model_name').map(key => (
                <Col span={12} key={key}>
                  <Form.Item name={key} label={key.replace(/_/g, ' ').toUpperCase()}>
                    <InputNumber style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              ))}
            </Row>
            <Button type="primary" htmlType="submit" loading={loading} block size="large" icon={<DashboardOutlined />}>
              PREDICT
            </Button>
          </Form>
        </Sider>

        <Content style={{ padding: '24px', background: '#f0f2f5' }}>
          <Row gutter={[24, 24]}>
            <Col span={8}>
              <Card><Statistic title="Current" value={initialValues.price_lag_0} suffix="VND" /></Card>
            </Col>
            <Col span={8}>
              <Card style={{ background: '#e6f7ff' }}>
                <Statistic title="AI Target" value={prediction ? prediction.predicted_price_tomorrow : 0} suffix="VND" />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic 
                    title="Change" 
                    value={prediction ? Math.abs(prediction.predicted_delta) : 0} 
                    valueStyle={{ color: prediction && prediction.predicted_delta > 0 ? '#cf1322' : '#3f8600' }}
                    prefix={prediction && prediction.predicted_delta > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                    suffix="VND" 
                />
              </Card>
            </Col>
            <Col span={24}>
              <Card title="Forecast Visualization">
                <div style={{ height: '350px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={['auto', 'auto']} />
                      <Tooltip />
                      <Area type="monotone" dataKey="price" stroke="#1890ff" fill="#e6f7ff" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </Col>
          </Row>
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
